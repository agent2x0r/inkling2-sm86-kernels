"""Authoritative PyTorch golden references for Thinking Machines' Inkling 2 Mini.

Every function in this module matches the exact numerics of vLLM PR #69028
(`vllm/models/inkling2/nvidia/ops/`) and Thinking Machines' reference implementation:

1. `NVFP4` expert weights (layers 0..19):
   - Packed `uint8` nibbles: low nibble `b & 0xF` is column `2*k`, high nibble `b >> 4`
     is column `2*k + 1`, decoded via IEEE-like FP4 `e2m1` (1 sign, 2 exp, 1 mantissa).
   - Multiplied by per-16-element `float8_e4m3fn` block scale and per-expert outer scale
     `amax[e] / outer_scale_denominator[0]` (where `outer_scale_denominator == 2688.0 == 6 * 448`).

2. `MXFP8` expert weights (layers 20..23):
   - `float8_e4m3fn` weights multiplied by `UE8M0` 32-element block scales (`2^(u8 - 127)`,
     with `u8 == 0` mapped to `2^-126` per OCP MX / `moe_sm120.py:_dequant_e4m3`).

3. `SiTU` ("Sigmoid-Tanh-Gated Linear Unit") expert activation (`ops/activation.py:_situ`):
   - `w13` output columns are INTERLEAVED `[gate_0, up_0, gate_1, up_1, ...]`!
   - `sigmoid` is computed on UN-CAPPED `gate` BEFORE softcapping `gate` at `4.0`
     and `up` at `25.0`:
         `y = ((4.0 * tanh(gate / 4.0)) * sigmoid(gate)) * (25.0 * tanh(up / 25.0)) * gamma`

4. Looped Core Cross-Pass Blend + RMSNorm (`ops/loop.py`):
   - Middle layers 6..17 run 3 passes (`48` logical visits from `24` physical layers).
   - Passes 1 and 2 blend `dest` and `message` in FP32 WITHOUT FMA contraction:
         `x = (1.0 - tanh(forget)) * dest + tanh(gain) * message`
     followed immediately by FP32 RMSNorm rounded once to BF16.

5. Sheared Relative-Bias Paged Attention (`ops/attention_sm90.py` + `ops/sconv.py`):
   - No RoPE. Each query projects `r` (`d_rel=16`) against `rel_logits_proj` (`[16, 1024]`).
   - `attention_prologue` writes `rel_logits` into a sheared `(T + 128, H, E + 256)` buffer
     aligned to 128-token query tiles, adds `bias` to `q @ k.T * scale` pre-softmax, applies
     causal + optional `W=1024` sliding-window mask, and post-multiplies by:
         `out * sigmoid(gate) * sigmoid(lse - sink)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Exact 4-bit e2m1 lookup table indexed by nibble 0..15 (bit 3 = sign, bits 2..1 = exp, bit 0 = mantissa)
E2M1_LUT: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class Inkling2MiniSpec:
    hidden_size: int = 2304
    intermediate_size: int = 1152
    num_physical_layers: int = 24
    loop_core_start: int = 6
    loop_core_end: int = 18
    loop_num_passes: int = 3
    num_q_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    sliding_window: int = 1024
    num_experts: int = 128
    top_k: int = 8
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    loop_gate_rank: int = 64
    sconv_kernel_size: int = 4
    checkpoint_bytes: int = 19_525_870_752  # Exact official safetensors byte count
    dspark_bytes: int = 671_088_640  # ~0.64 GB MTP/DSpark weights

    @property
    def gqa_group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads  # 3

    @property
    def num_layer_visits(self) -> int:
        core = self.loop_core_end - self.loop_core_start
        non_core = self.num_physical_layers - core
        return non_core + core * self.loop_num_passes  # 12 + 12 * 3 = 48

    @property
    def num_full_attn_visits(self) -> int:
        # Physical full-attention layers are 5 (prelude), 11 & 17 (core x3), 23 (coda)
        return 1 + 2 * self.loop_num_passes + 1  # 8

    @property
    def num_swa_attn_visits(self) -> int:
        return self.num_layer_visits - self.num_full_attn_visits  # 40


INKLING2_MINI_SPEC = Inkling2MiniSpec()


@dataclass(frozen=True)
class VRAMBudget:
    tp_size: int
    context_len: int
    include_dspark: bool
    kv_dtype_bytes: int
    weights_per_gpu_gib: float
    full_kv_per_gpu_gib: float
    swa_kv_per_gpu_gib: float
    conv_cache_per_gpu_gib: float
    total_static_plus_kv_gib: float
    headroom_on_24gib_card_gib: float


def estimate_vram_budget(
    context_len: int,
    tp_size: int = 1,
    *,
    include_dspark: bool = True,
    kv_dtype_bytes: int = 2,
    spec: Inkling2MiniSpec = INKLING2_MINI_SPEC,
) -> VRAMBudget:
    """Compute exact per-GPU static weight + 48-visit KV/conv cache footprint on 24 GiB 3090s.

    Crucial architectural insight:
    40 of Inkling 2 Mini's 48 layer visits are sliding-window (`W = 1024`), so their KV caches
    cap at 1,024 tokens regardless of context length! Only 8 visits (layers 5, 11x3, 17x3, 23)
    grow linearly with `context_len`.
    """
    if tp_size not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported tp_size={tp_size}")
    gib = float(1024**3)

    total_weights = spec.checkpoint_bytes + (spec.dspark_bytes if include_dspark else 0)
    # Embeddings/norms/small replicated buffers add ~0.25 GiB overhead on TP>1
    replicated_overhead = 0.25 * gib if tp_size > 1 else 0.0
    weights_per_gpu = (total_weights / tp_size + replicated_overhead) / gib

    kv_heads_per_gpu = spec.num_kv_heads // tp_size
    bytes_per_tok_per_visit = kv_heads_per_gpu * spec.head_dim * 2 * kv_dtype_bytes

    full_kv_bytes = spec.num_full_attn_visits * context_len * bytes_per_tok_per_visit
    swa_tokens = min(context_len, spec.sliding_window)
    swa_kv_bytes = spec.num_swa_attn_visits * swa_tokens * bytes_per_tok_per_visit

    # 48 conv-state caches: (K, V sharded; attn, mlp replicated), kernel window = 3 past taps
    conv_width_per_visit = (2 * kv_heads_per_gpu * spec.head_dim) + (2 * spec.hidden_size)
    conv_bytes = spec.num_layer_visits * 4 * conv_width_per_visit * 2  # negligible (~2 MiB/seq)

    total_gib = weights_per_gpu + (full_kv_bytes + swa_kv_bytes + conv_bytes) / gib
    return VRAMBudget(
        tp_size=tp_size,
        context_len=context_len,
        include_dspark=include_dspark,
        kv_dtype_bytes=kv_dtype_bytes,
        weights_per_gpu_gib=weights_per_gpu,
        full_kv_per_gpu_gib=full_kv_bytes / gib,
        swa_kv_per_gpu_gib=swa_kv_bytes / gib,
        conv_cache_per_gpu_gib=conv_bytes / gib,
        total_static_plus_kv_gib=total_gib,
        headroom_on_24gib_card_gib=24.0 - total_gib,
    )


def dequant_nvfp4_weight(
    packed_u8: torch.Tensor,
    block_scale_e4m3: torch.Tensor,
    amax_f32: torch.Tensor | None = None,
    denom_f32: torch.Tensor | float = 2688.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed NVFP4 `[..., N, K // 2]` + E4M3 `[..., N, K // 16]` -> `[..., N, K]`.

    Matches `tests/models/inkling2/test_moe.py:dequant_nvfp4` and `Inkling2Experts.scales`.
    """
    if packed_u8.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 packed NVFP4, got {packed_u8.dtype}")
    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=packed_u8.device)
    lo = lut[(packed_u8 & 0x0F).long()]
    hi = lut[(packed_u8 >> 4).long()]
    unpacked = torch.stack((lo, hi), dim=-1).flatten(-2)  # [..., N, K]

    scales = block_scale_e4m3.to(torch.float32).repeat_interleave(16, dim=-1)
    val = unpacked * scales
    if amax_f32 is not None:
        denom = denom_f32.to(torch.float32) if isinstance(denom_f32, torch.Tensor) else float(denom_f32)
        outer = (amax_f32.to(torch.float32) / denom).view(*amax_f32.shape, 1, 1)
        val = val * outer
    return val.to(out_dtype)


def dequant_mxfp8_weight(
    weight_e4m3: torch.Tensor,
    block_scale_ue8m0: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP8 E4M3 `[..., N, K]` + UE8M0 `[..., N, K // 32]` -> `[..., N, K]`.

    Matches `vllm/models/inkling2/nvidia/ops/moe_sm120.py:_dequant_e4m3`:
    `exponents == 0` maps to IEEE float32 bits `0x00400000` (`2^-126`), otherwise `exponents << 23`
    (`2^(e - 127)`).
    """
    if block_scale_ue8m0.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 UE8M0 scales, got {block_scale_ue8m0.dtype}")
    exp_u32 = block_scale_ue8m0.to(torch.int32)
    bits = torch.where(exp_u32 == 0, 0x00400000, exp_u32 << 23)
    scales = bits.view(torch.float32).repeat_interleave(32, dim=-1)
    return (weight_e4m3.to(torch.float32) * scales).to(out_dtype)


def situ_activation_ref(
    gate_up: torch.Tensor,
    routing_weight: torch.Tensor | float = 1.0,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Exact Inkling 2 `SiTU` expert activation on interleaved `[..., 2 * inter]` -> `[..., inter]`.

    CRITICAL detail from `ops/activation.py:_situ`:
    1. Even columns `0::2` are `gate`, odd columns `1::2` are `up`.
    2. `sigmoid(gate)` is evaluated BEFORE `gate` is softcapped by `gate_cap * tanh(gate / gate_cap)`.
    """
    gate = gate_up[..., 0::2].to(torch.float32)
    up = gate_up[..., 1::2].to(torch.float32)
    sig = torch.sigmoid(gate)
    gate_capped = gate_cap * torch.tanh(gate / gate_cap) if gate_cap > 0 else gate
    up_capped = up_cap * torch.tanh(up / up_cap) if up_cap > 0 else up
    if isinstance(routing_weight, torch.Tensor):
        gamma = routing_weight.to(torch.float32).unsqueeze(-1)
    else:
        gamma = float(routing_weight)
    return (((gate_capped * sig) * up_capped) * gamma).to(torch.bfloat16)


def expert_mlp_ref(
    x_bf16: torch.Tensor,
    w13_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    routing_weight: torch.Tensor | float = 1.0,
    gate_cap: float = 4.0,
    up_cap: float = 25.0,
) -> torch.Tensor:
    """Single-expert forward reference: `x [T, H]` -> `w13 [2*I, H]` -> `SiTU` -> `w2 [H, I]`."""
    gate_up = (x_bf16.float() @ w13_bf16.float().T).to(torch.bfloat16)
    act = situ_activation_ref(gate_up, routing_weight, gate_cap=gate_cap, up_cap=up_cap)
    down = act.float() @ w2_bf16.float().T
    return down.to(torch.bfloat16)


def loop_pair_rms_norm_ref(
    dest: torch.Tensor,
    message: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference for `ops/loop.py:pair_rms_norm` (`(S, T, D)` fp32 -> `(S, T, 2*D)` bf16)."""
    d = message.shape[-1]
    dest_f = dest.to(torch.float32).expand_as(message)
    msg_f = message.to(torch.float32)
    dest_norm = (dest_f * torch.rsqrt((dest_f * dest_f).sum(dim=-1, keepdim=True) / d + eps)).to(
        torch.bfloat16
    )
    msg_norm = (msg_f * torch.rsqrt((msg_f * msg_f).sum(dim=-1, keepdim=True) / d + eps)).to(
        torch.bfloat16
    )
    return torch.cat((dest_norm, msg_norm), dim=-1)


def loop_blend_rms_norm_ref(
    dest: torch.Tensor,
    message: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    """Reference for `ops/loop.py:blend_rms_norm` with non-contracted FP32 blend.

    `gain_forget` has shape `(S, T, 2*D)` packed as `[gain, forget]`.
    `x = (1 - tanh(forget)) * dest + tanh(gain) * message` (two separate FP32 products + FP32 add),
    then RMSNorm with `norm_weight (S, D)` -> BF16 `(S, T, D)`.
    """
    d = message.shape[-1]
    dest_f = dest.to(torch.float32).expand_as(message)
    msg_f = message.to(torch.float32)
    gf_f = gain_forget.to(torch.float32)
    gain = torch.tanh(gf_f[..., :d])
    forget = torch.tanh(gf_f[..., d:])
    # Separate products prevent FMA contraction in Python/ATen float32 tensors
    term_keep = (1.0 - forget) * dest_f
    term_msg = gain * msg_f
    blended = term_keep + term_msg
    rstd = torch.rsqrt((blended * blended).sum(dim=-1, keepdim=True) / d + norm_eps)
    normed = (blended * rstd) * norm_weight.to(torch.float32).unsqueeze(1)
    return normed.to(torch.bfloat16)


def shift_rel_proj_ref(rel_proj: torch.Tensor, is_local: bool) -> torch.Tensor:
    """Exact copy of `ops/sconv.py:shift_rel_proj` (`(d_rel, E)` -> `(8, d_rel + 1, E + 512)`)."""
    shift_pad = 512
    _, extent = rel_proj.shape
    shift = torch.arange(8, device=rel_proj.device)[:, None]
    y = torch.arange(extent + shift_pad, device=rel_proj.device)
    dist = shift + extent + 256 - y
    outside = (dist < 0) | (dist >= extent)
    proj = rel_proj[:, dist.clamp(0, extent - 1)].transpose(0, 1)
    pad = (dist >= extent) if is_local else torch.zeros_like(outside)
    pad = (dist < 0) | pad
    pad_row = torch.zeros_like(proj[:, :1]).masked_fill(pad[:, None], float("-inf"))
    return torch.cat([proj.masked_fill(outside[:, None], 0), pad_row], dim=1).contiguous()


def shear_rel_logits_ref(
    rel_logits: torch.Tensor,
    seqs: list[tuple[int, int]],
    tile: int = 128,
    is_local: bool = True,
) -> torch.Tensor:
    """Exact reference of `attention_prologue`'s sheared relative-bias layout.

    Given natural `rel_logits` `[T, H, E]` where `rel_logits[t, h, d]` is the bias at distance
    `d = q_pos - k_pos` (`0 <= d < E`), lays out `(T + 128, H, E + 256)` so that inside each
    128-token query tile ending at `tile_end`, column `c` aligns directly with key index:
        `k_idx = c + cdiv(past + tile_end, 128) * 128 - (E + 256)`.
    """
    tokens, heads, extent = rel_logits.shape
    out = rel_logits.new_full((tokens + 128, heads, extent + 256), 0.0)
    cols = torch.arange(extent + 256, device=rel_logits.device)
    start = 0
    for q_len, kv_len in seqs:
        idx = torch.arange(q_len, device=rel_logits.device)
        pos = kv_len - q_len + idx
        tile_end = ((idx // tile + 1) * tile).clamp(max=q_len)
        n_max = (kv_len - q_len + tile_end + 127) // 128
        # Note: extent + 256 == 10 * 128 when extent == 1024; general form uses -(extent + 256)
        key_base = n_max[:, None] * 128 - (extent + 256)
        dist = pos[:, None] - (cols[None, :] + key_base)
        valid = (dist >= 0) & (dist < extent)
        rel = rel_logits[start : start + q_len]
        vals = rel.gather(2, dist.clamp(0, extent - 1)[:, None, :].expand(-1, heads, -1))
        pad = torch.where((dist < 0) | is_local, float("-inf"), 0.0).to(rel_logits.dtype)
        out[start : start + q_len] = torch.where(valid[:, None, :], vals, pad[:, None, :])
        start += q_len
    return out


def paged_rel_attention_ref(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits_natural: torch.Tensor,
    seqs: list[tuple[int, int]],
    block_table: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    window_size: int = 0,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 golden oracle for Inkling 2 Mini paged GQA with relative bias, SWA, gate, and sink.

    Args:
        q: `[total_q, H_q, D]` BF16/FP32 queries (`H_q=24, D=128`).
        key_cache, value_cache: `[num_blocks, page_size, H_kv, D]` (`H_kv=8`).
        rel_logits_natural: `[total_q, H_q, rel_extent]` natural distance-indexed bias (`E=1024`).
        seqs: list of `(q_len, kv_len)` per request in batch.
        block_table: `[num_reqs, max_blocks]` int32 page table.
        window_size: `1024` for sliding-window layers (keys `[pos - 1023 .. pos]`), `0` for full.
        gate: optional `[total_q, H_q * D]` post-attention gate logits.
        sinks: optional `[total_q, H_q]` per-token attention sink logits.

    Returns:
        `(out, lse)` where `out` is `[total_q, H_q, D]` (FP32 if `gate is None`, else BF16 gated)
        and `lse` is `[H_q, total_q]` natural-log LSE.
    """
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = key_cache.shape[2]
    group = num_heads // num_kv_heads
    extent = rel_logits_natural.shape[2]
    scale = (head_dim**-0.5) if softmax_scale is None else softmax_scale

    out = torch.empty((total_q, num_heads, head_dim), dtype=torch.float32, device=q.device)
    lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)

    start = 0
    for req_idx, (q_len, kv_len) in enumerate(seqs):
        pages = block_table[req_idx].long()
        k = (
            key_cache[pages]
            .flatten(0, 1)[:kv_len]
            .to(torch.float32)
            .repeat_interleave(group, dim=1)
        )
        v = (
            value_cache[pages]
            .flatten(0, 1)[:kv_len]
            .to(torch.float32)
            .repeat_interleave(group, dim=1)
        )
        q_slice = q[start : start + q_len].to(torch.float32)
        q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        k_pos = torch.arange(kv_len, device=q.device)
        dist = q_pos[:, None] - k_pos[None, :]  # [q_len, kv_len]

        rel = rel_logits_natural[start : start + q_len].to(torch.float32).permute(1, 0, 2)
        gather_idx = dist.clamp(0, extent - 1).unsqueeze(0).expand(num_heads, -1, -1)
        in_extent = (dist >= 0) & (dist < extent)
        bias = rel.gather(2, gather_idx).masked_fill(~in_extent.unsqueeze(0), 0.0)

        scores = torch.einsum("qhd,khd->hqk", q_slice, k) * scale + bias
        visible = dist >= 0
        if window_size > 0:
            visible = visible & (dist < window_size)
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))

        req_lse = torch.logsumexp(scores, dim=-1)  # [H_q, q_len]
        # Guard all-masked rows (if any) from NaN
        probs = torch.where(
            visible.unsqueeze(0),
            torch.exp(scores - req_lse.masked_fill(req_lse == float("-inf"), 0.0).unsqueeze(-1)),
            0.0,
        )
        lse[:, start : start + q_len] = req_lse
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", probs, v)
        start += q_len

    if gate is not None and sinks is not None:
        gate_f = gate.view(total_q, num_heads, head_dim).to(torch.float32)
        sink_f = sinks.view(total_q, num_heads).to(torch.float32)
        sink_scale = torch.sigmoid(lse.T - sink_f).unsqueeze(-1)
        gated = out * torch.sigmoid(gate_f) * sink_scale
        return gated.to(torch.bfloat16), lse

    return out, lse
