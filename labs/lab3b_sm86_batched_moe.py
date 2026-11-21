"""Lab 3B (Exercise 3B): Batched Top-8 SM86 MoE (`NVFP4` & `MXFP8`) — Warp-GEMV vs Grouped Tensor Core.

Why Exercise 3B matters on an RTX 3090 (SM86, 82 SMs, 840 GB/s GDDR6X):
In Exercise 3A, a single-expert NVFP4 `w13` launch (`2.99 MB`) reached `7.68 us` (`389 GB/s`) —
3.19x faster than dense BF16 cuBLAS and 1.41x faster than Humming — yet still < 50% of GA102's
`840 GB/s` peak because 1 expert only launches 72 CTAs (< 1 wave) and pays separate `SiTU` +
Top-8 reduction DRAM round-trips.

In real Inkling-2-Mini decode (`T` tokens, `E = 128` experts, `Top-K = 8`):
  - Layers 0..19 (`NVFP4`, 44 of 48 logical layer visits!):
      `w13` reads `8 x 2.99 MB = 23.89 MB` -> SOL @ 840 GB/s = `28.4 us`
      `w2`  reads `8 x 1.49 MB = 11.94 MB` -> SOL @ 840 GB/s = `14.2 us`
      Total 1-layer NVFP4 MoE SOL (`T = 1`) = **`42.6 us`**!
  - Layers 20..23 (`MXFP8`, 4 of 48 logical layer visits):
      `w13` reads `8 x 5.47 MB = 43.79 MB` -> SOL @ 840 GB/s = `52.1 us`
      `w2`  reads `8 x 2.74 MB = 21.90 MB` -> SOL @ 840 GB/s = `26.1 us`
      Total 1-layer MXFP8 MoE SOL (`T = 1`) = **`78.2 us`**!

This lab builds and benchmarks TWO complementary SM86 MoE engines side-by-side against Dense BF16
and Production Humming (`GemmType.INDEXED`) on **8 real Inkling-2-Mini Layer 0 NVFP4 experts**
and **8 real Layer 20 MXFP8 experts** across `T = 1, 4, 16, 64` tokens:

  1. **CUDA C++ Warp-GEMV Decode Engine (`csrc/nvfp4_moe_sm86.cu`)** for `T = 1..4`:
     - 1 thread = 1 exact quantization scale group (16 weights `uint2` for NVFP4; 32 weights
       `2x uint4` for MXFP8) -> 16x/32x fewer scale multiplies & zero lane tail divergence!
     - Fused in-register exact `SiTU` epilogue (zero `gate_up` DRAM write, zero `_situ_kernel`!).
     - Fused zero-`atomicAdd` Top-8 `w2` reduction (plus an expert-parallel atomic variant to race!).
  2. **Triton Grouped Tensor Core `lop3` MoE Engine** for `T = 8..256`:
     - Uses vLLM-standard `moe_align_block_size` (`sorted_ids`, `expert_ids`, `num_tokens_padded`).
     - Combines Exercise 3A's 96-reg `lop3` decode with a **fused `SiTU` epilogue in registers**
       (`tl.split(tl.reshape(acc, (BM, BN // 2, 2)))`) and FP32 `tl.atomic_add` `w2` reduction!
  3. **Automatic Dispatcher**: Demonstrates the real-silicon crossover batch size `T*` where
     memory-bound Warp-GEMV hands off to compute-bound Grouped Tensor Cores!
"""

from __future__ import annotations

from pathlib import Path

import torch

from inkling_sm86.checkpoint_slice import (
    fetch_real_mxfp8_expert_sample,
    fetch_real_nvfp4_expert_sample,
)
from inkling_sm86.reference import (
    dequant_mxfp8_weight,
    dequant_nvfp4_weight,
    situ_activation_ref,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def choose_moe_block_m(num_tokens: int, top_k: int, num_experts: int) -> int:
    """Choose `block_m` maximizing SM86 occupancy & weight reuse across active experts.

    Lesson 3B real-silicon finding on RTX 3090 (GA102):
    - `avg_m <= 16`: `BM = 16` (`BN=32, BK=256`, 1 pass over weights, ~66 us NVFP4 / 93 us MXFP8)
    - `avg_m > 16`:  `BM = 32` (`BN=64, BK=128`, 108 regs -> 3 CTAs/SM, 179 us at T=64!), which
      beats `BM = 64` (148 regs -> 1 CTA/SM, 285 us!) by 1.59x on SM86!
    """
    avg_m = (num_tokens * top_k + num_experts - 1) // num_experts
    return 16 if avg_m <= 16 else 32


def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_m: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group `(T, TOP_K)` expert assignments into `block_m`-aligned tiles (vLLM / Humming contract).

    Returns:
      - `sorted_ids`: `[num_padded]` int32 flat slot index `a = tok * TOP_K + k`, or sentinel
        `T * TOP_K` for padding rows.
      - `expert_ids`: `[num_m_blocks]` int32 expert index for each `block_m` tile.
      - `num_tokens_padded`: `[1]` int32 total padded rows (`num_m_blocks * block_m`).
    """
    flat_ids = topk_ids.reshape(-1)
    total_slots = flat_ids.numel()
    device = topk_ids.device

    sorted_chunks: list[torch.Tensor] = []
    expert_chunks: list[int] = []
    for e in range(num_experts):
        slots = torch.nonzero(flat_ids == e, as_tuple=False).flatten().to(torch.int32)
        count = slots.numel()
        if count == 0:
            continue
        num_blocks = (count + block_m - 1) // block_m
        padded_len = num_blocks * block_m
        if padded_len > count:
            pad = torch.full((padded_len - count,), total_slots, dtype=torch.int32, device=device)
            slots = torch.cat((slots, pad), dim=0)
        sorted_chunks.append(slots)
        expert_chunks.extend([e] * num_blocks)

    sorted_ids = torch.cat(sorted_chunks, dim=0).contiguous()
    expert_ids = torch.tensor(expert_chunks, dtype=torch.int32, device=device)
    num_padded = torch.tensor([sorted_ids.numel()], dtype=torch.int32, device=device)
    return sorted_ids, expert_ids, num_padded


def batched_moe_ref(
    x: torch.Tensor,            # [T, H] bf16
    w13_bf16: torch.Tensor,     # [E, 2*I, H] bf16
    w2_bf16: torch.Tensor,      # [E, H, I] bf16
    topk_ids: torch.Tensor,     # [T, TOP_K] int32/int64
    topk_weights: torch.Tensor, # [T, TOP_K] fp32
) -> torch.Tensor:
    """Authoritative FP32-accumulated golden reference for Top-K Inkling-2-Mini MoE."""
    tokens, hidden = x.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros((tokens, hidden), dtype=torch.float32, device=x.device)
    for t in range(tokens):
        xt = x[t : t + 1].float()
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            gamma = topk_weights[t, k : k + 1]
            gate_up = (xt @ w13_bf16[e].float().T).to(torch.bfloat16)
            act = situ_activation_ref(gate_up, gamma)
            down = act.float() @ w2_bf16[e].float().T
            out[t] += down[0]
    return out.to(torch.bfloat16)


def make_synthetic_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate realistic Top-K routing (distinct experts per token, realistic softmax weights)."""
    assert top_k <= num_experts
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    top_vals, top_idx = torch.topk(logits, k=top_k, dim=-1)
    weights = torch.softmax(top_vals, dim=-1).contiguous()
    return top_idx.to(torch.int32).contiguous(), weights


def load_cuda_moe_extension():
    """JIT-compile `csrc/nvfp4_moe_sm86.cu` WITHOUT `--use_fast_math` (keeps non-FTZ & exact tanh!)."""
    from torch.utils.cpp_extension import load

    src = REPO_ROOT / "csrc" / "nvfp4_moe_sm86.cu"
    return load(
        name="inkling2_nvfp4_moe_sm86",
        sources=[str(src)],
        extra_cuda_cflags=[
            "-O3",
            "-arch=sm_86",
            "--ptxas-options=-v",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        verbose=False,
    )


def build_sm86_batched_moe_triton_kernels():
    """Build grouped Tensor Core `lop3` MoE kernels (NVFP4 + MXFP8) with fused `SiTU` epilogue."""
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _unpack_e2m1_pair_lop3(packed_u8):
        u = packed_u8.to(tl.uint32)
        even_bits = ((u & 0x08) << 28) | ((u & 0x07) << 22)
        odd_bits = ((u & 0x80) << 24) | ((u & 0x70) << 18)
        return even_bits.to(tl.float32, bitcast=True), odd_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _decode_e4m3_scaled_2p118(u8_val):
        u = u8_val.to(tl.uint32)
        raw = (((u & 0x80) << 24) | ((u & 0x7F) << 20)).to(tl.float32, bitcast=True)
        return (raw * 1.3292279957849159e36) * 3.3230699894622897e35

    @triton.jit
    def _nvfp4_moe_w13_situ_lop3_kernel(
        X,             # [T, K] bf16
        W13_packed,    # [E, 2*N_INTER, K//2] uint8
        W13_scale,     # [E, 2*N_INTER, K//16] uint8
        W13_outer,     # [E] fp32
        TopkWeights,   # [T * TOP_K] fp32
        SortedIds,     # [num_padded] int32
        ExpertIds,     # [num_m_blocks] int32
        ActOut,        # [T * TOP_K, N_INTER] bf16
        total_slots,
        N: tl.constexpr,        # 2 * N_INTER (2304)
        N_INTER: tl.constexpr,  # 1152
        K: tl.constexpr,        # 2304
        TOP_K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,       # even! (32 -> 16 gate/up pairs)
        BK: tl.constexpr,       # 256
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        expert_id = tl.load(ExpertIds + pid_m).to(tl.int64)
        a_ids = tl.load(SortedIds + pid_m * BM + tl.arange(0, BM))
        valid = a_ids < total_slots
        tok_ids = (a_ids // TOP_K).to(tl.int64)

        cols = pid_n * BN + tl.arange(0, BN)
        col_mask = cols < N

        w_expert_base = expert_id * (N * (K // 2))
        s_expert_base = expert_id * (N * (K // 16))

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x_tile = tl.load(
                X + tok_ids[:, None] * K + offs_k[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            offs_kp = (k0 // 2) + tl.arange(0, BK // 2)
            w_u8 = tl.load(
                W13_packed + w_expert_base + cols[:, None].to(tl.int64) * (K // 2) + offs_kp[None, :],
                mask=col_mask[:, None],
                other=0,
            )
            offs_ks = (k0 // 16) + tl.arange(0, BK // 16)
            s_u8 = tl.load(
                W13_scale + s_expert_base + cols[:, None].to(tl.int64) * (K // 16) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            )

            w_even, w_odd = _unpack_e2m1_pair_lop3(w_u8)
            s_f32 = _decode_e4m3_scaled_2p118(s_u8)
            s_half = tl.reshape(
                tl.broadcast_to(s_f32[:, :, None], (BN, BK // 16, 8)),
                (BN, BK // 2),
            )
            w_bf16 = tl.interleave((w_even * s_half).to(tl.bfloat16), (w_odd * s_half).to(tl.bfloat16))
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc)

        # Fused in-register SiTU epilogue!
        outer = tl.load(W13_outer + expert_id) * 256.0
        acc = acc * outer
        gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
        sig = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        g_cap = 4.0 * libdevice.tanh(gate * 0.25)
        u_cap = 25.0 * libdevice.tanh(up * 0.04)
        gamma = tl.load(TopkWeights + a_ids, mask=valid, other=0.0).to(tl.float32)
        y = ((g_cap * sig) * u_cap) * gamma[:, None]

        inter_cols = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        inter_mask = inter_cols < N_INTER
        tl.store(
            ActOut + a_ids[:, None].to(tl.int64) * N_INTER + inter_cols[None, :],
            y.to(tl.bfloat16),
            mask=valid[:, None] & inter_mask[None, :],
        )

    @triton.jit
    def _nvfp4_moe_w2_lop3_kernel(
        Act,           # [T * TOP_K, K] bf16
        W2_packed,     # [E, N, K//2] uint8
        W2_scale,      # [E, N, K//16] uint8
        W2_outer,      # [E] fp32
        SortedIds,     # [num_padded] int32
        ExpertIds,     # [num_m_blocks] int32
        OutF32,        # [T, N] fp32 (zero-initialized, reduced via L2 atomic_add!)
        total_slots,
        N: tl.constexpr,  # H = 2304
        K: tl.constexpr,  # I = 1152
        TOP_K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,  # 128 (divides 1152 evenly: 9 iters!)
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        expert_id = tl.load(ExpertIds + pid_m).to(tl.int64)
        a_ids = tl.load(SortedIds + pid_m * BM + tl.arange(0, BM))
        valid = a_ids < total_slots
        tok_ids = (a_ids // TOP_K).to(tl.int64)

        cols = pid_n * BN + tl.arange(0, BN)
        col_mask = cols < N

        w_expert_base = expert_id * (N * (K // 2))
        s_expert_base = expert_id * (N * (K // 16))

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            a_tile = tl.load(
                Act + a_ids[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            offs_kp = (k0 // 2) + tl.arange(0, BK // 2)
            w_u8 = tl.load(
                W2_packed + w_expert_base + cols[:, None].to(tl.int64) * (K // 2) + offs_kp[None, :],
                mask=col_mask[:, None],
                other=0,
            )
            offs_ks = (k0 // 16) + tl.arange(0, BK // 16)
            s_u8 = tl.load(
                W2_scale + s_expert_base + cols[:, None].to(tl.int64) * (K // 16) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            )

            w_even, w_odd = _unpack_e2m1_pair_lop3(w_u8)
            s_f32 = _decode_e4m3_scaled_2p118(s_u8)
            s_half = tl.reshape(
                tl.broadcast_to(s_f32[:, :, None], (BN, BK // 16, 8)),
                (BN, BK // 2),
            )
            w_bf16 = tl.interleave((w_even * s_half).to(tl.bfloat16), (w_odd * s_half).to(tl.bfloat16))
            acc = tl.dot(a_tile, tl.trans(w_bf16), acc)

        outer = tl.load(W2_outer + expert_id) * 256.0
        acc = acc * outer
        tl.atomic_add(
            OutF32 + tok_ids[:, None] * N + cols[None, :],
            acc,
            mask=valid[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _mxfp8_moe_w13_situ_kernel(
        X,             # [T, K] bf16
        W13_e4m3,      # [E, 2*N_INTER, K] uint8
        W13_ue8m0,     # [E, 2*N_INTER, K//32] uint8
        TopkWeights,   # [T * TOP_K] fp32
        SortedIds,     # [num_padded] int32
        ExpertIds,     # [num_m_blocks] int32
        ActOut,        # [T * TOP_K, N_INTER] bf16
        total_slots,
        N: tl.constexpr,
        N_INTER: tl.constexpr,
        K: tl.constexpr,
        TOP_K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,  # 128
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        expert_id = tl.load(ExpertIds + pid_m).to(tl.int64)
        a_ids = tl.load(SortedIds + pid_m * BM + tl.arange(0, BM))
        valid = a_ids < total_slots
        tok_ids = (a_ids // TOP_K).to(tl.int64)

        cols = pid_n * BN + tl.arange(0, BN)
        col_mask = cols < N

        w_expert_base = expert_id * (N * K)
        s_expert_base = expert_id * (N * (K // 32))

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x_tile = tl.load(
                X + tok_ids[:, None] * K + offs_k[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            w_u8 = tl.load(
                W13_e4m3 + w_expert_base + cols[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=col_mask[:, None],
                other=0,
            ).to(tl.uint32)
            offs_ks = (k0 // 32) + tl.arange(0, BK // 32)
            exp_u8 = tl.load(
                W13_ue8m0 + s_expert_base + cols[:, None].to(tl.int64) * (K // 32) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            ).to(tl.uint32)

            # Fold e4m3fn's 2^+120 exponent bias directly into UE8M0 (single FP32 mul per weight!)
            w_raw = (((w_u8 & 0x80) << 24) | ((w_u8 & 0x7F) << 20)).to(tl.float32, bitcast=True)
            exp_eff = tl.where(exp_u8 == 0, 1, exp_u8)
            s_bits = (exp_eff + 120) << 23
            s_f32 = s_bits.to(tl.float32, bitcast=True)
            s_exp = tl.reshape(tl.broadcast_to(s_f32[:, :, None], (BN, BK // 32, 32)), (BN, BK))
            w_bf16 = (w_raw * s_exp).to(tl.bfloat16)
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc)

        gate, up = tl.split(tl.reshape(acc, (BM, BN // 2, 2)))
        sig = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        g_cap = 4.0 * libdevice.tanh(gate * 0.25)
        u_cap = 25.0 * libdevice.tanh(up * 0.04)
        gamma = tl.load(TopkWeights + a_ids, mask=valid, other=0.0).to(tl.float32)
        y = ((g_cap * sig) * u_cap) * gamma[:, None]

        inter_cols = pid_n * (BN // 2) + tl.arange(0, BN // 2)
        inter_mask = inter_cols < N_INTER
        tl.store(
            ActOut + a_ids[:, None].to(tl.int64) * N_INTER + inter_cols[None, :],
            y.to(tl.bfloat16),
            mask=valid[:, None] & inter_mask[None, :],
        )

    @triton.jit
    def _mxfp8_moe_w2_kernel(
        Act,           # [T * TOP_K, K] bf16
        W2_e4m3,       # [E, N, K] uint8
        W2_ue8m0,      # [E, N, K//32] uint8
        SortedIds,     # [num_padded] int32
        ExpertIds,     # [num_m_blocks] int32
        OutF32,        # [T, N] fp32
        total_slots,
        N: tl.constexpr,
        K: tl.constexpr,
        TOP_K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,  # 128
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        expert_id = tl.load(ExpertIds + pid_m).to(tl.int64)
        a_ids = tl.load(SortedIds + pid_m * BM + tl.arange(0, BM))
        valid = a_ids < total_slots
        tok_ids = (a_ids // TOP_K).to(tl.int64)

        cols = pid_n * BN + tl.arange(0, BN)
        col_mask = cols < N

        w_expert_base = expert_id * (N * K)
        s_expert_base = expert_id * (N * (K // 32))

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            a_tile = tl.load(
                Act + a_ids[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            w_u8 = tl.load(
                W2_e4m3 + w_expert_base + cols[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=col_mask[:, None],
                other=0,
            ).to(tl.uint32)
            offs_ks = (k0 // 32) + tl.arange(0, BK // 32)
            exp_u8 = tl.load(
                W2_ue8m0 + s_expert_base + cols[:, None].to(tl.int64) * (K // 32) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            ).to(tl.uint32)

            w_raw = (((w_u8 & 0x80) << 24) | ((w_u8 & 0x7F) << 20)).to(tl.float32, bitcast=True)
            exp_eff = tl.where(exp_u8 == 0, 1, exp_u8)
            s_f32 = ((exp_eff + 120) << 23).to(tl.float32, bitcast=True)
            s_exp = tl.reshape(tl.broadcast_to(s_f32[:, :, None], (BN, BK // 32, 32)), (BN, BK))
            w_bf16 = (w_raw * s_exp).to(tl.bfloat16)
            acc = tl.dot(a_tile, tl.trans(w_bf16), acc)

        tl.atomic_add(
            OutF32 + tok_ids[:, None] * N + cols[None, :],
            acc,
            mask=valid[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _cast_and_zero_kernel(BufF32, OutBF16, n_elements, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        v = tl.load(BufF32 + offs, mask=mask, other=0.0)
        tl.store(OutBF16 + offs, v.to(tl.bfloat16), mask=mask)
        tl.store(BufF32 + offs, 0.0, mask=mask)

    def run_nvfp4_grouped_moe(
        x: torch.Tensor,
        w13_packed: torch.Tensor,
        w13_scale: torch.Tensor,
        w13_outer: torch.Tensor,
        w2_packed: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_outer: torch.Tensor,
        topk_weights: torch.Tensor,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        top_k: int,
        block_m: int = 16,
        act_buf: torch.Tensor | None = None,
        f32_buf: torch.Tensor | None = None,
        out_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, k = x.shape
        n_w13 = w13_packed.shape[1]
        n_inter = n_w13 // 2
        h = w2_packed.shape[1]
        total_slots = tokens * top_k
        num_m_blocks = expert_ids.numel()

        if act_buf is None:
            act_buf = torch.empty((total_slots, n_inter), dtype=torch.bfloat16, device=x.device)
        if f32_buf is None:
            f32_buf = torch.zeros((tokens, h), dtype=torch.float32, device=x.device)
        if out_buf is None:
            out_buf = torch.empty((tokens, h), dtype=torch.bfloat16, device=x.device)

        if block_m >= 64:
            bn, bk1, bk2, nw, ns = 64, 128, 128, 8, 2
        elif block_m >= 32:
            bn, bk1, bk2, nw, ns = 64, 128, 128, 4, 3
        else:
            bn = 32
            bk1 = 256 if (k % 256 == 0) else 128
            bk2 = 128 if (n_inter % 128 == 0) else 64
            nw, ns = 4, 3

        c1 = _nvfp4_moe_w13_situ_lop3_kernel[(num_m_blocks, triton.cdiv(n_w13, bn))](
            x,
            w13_packed,
            w13_scale.view(torch.uint8),
            w13_outer,
            topk_weights.reshape(-1),
            sorted_ids,
            expert_ids,
            act_buf,
            total_slots,
            N=n_w13,
            N_INTER=n_inter,
            K=k,
            TOP_K=top_k,
            BM=block_m,
            BN=bn,
            BK=bk1,
            num_warps=nw,
            num_stages=ns,
        )
        c2 = _nvfp4_moe_w2_lop3_kernel[(num_m_blocks, triton.cdiv(h, bn))](
            act_buf,
            w2_packed,
            w2_scale.view(torch.uint8),
            w2_outer,
            sorted_ids,
            expert_ids,
            f32_buf,
            total_slots,
            N=h,
            K=n_inter,
            TOP_K=top_k,
            BM=block_m,
            BN=bn,
            BK=bk2,
            num_warps=nw,
            num_stages=ns,
        )
        n_el = tokens * h
        _cast_and_zero_kernel[(triton.cdiv(n_el, 512),)](f32_buf, out_buf, n_el, BLOCK=512, num_warps=4)
        run_nvfp4_grouped_moe.last_compiled = (c1, c2)  # type: ignore[attr-defined]
        return out_buf

    def run_mxfp8_grouped_moe(
        x: torch.Tensor,
        w13_e4m3: torch.Tensor,
        w13_ue8m0: torch.Tensor,
        w2_e4m3: torch.Tensor,
        w2_ue8m0: torch.Tensor,
        topk_weights: torch.Tensor,
        sorted_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        top_k: int,
        block_m: int = 16,
        act_buf: torch.Tensor | None = None,
        f32_buf: torch.Tensor | None = None,
        out_buf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, k = x.shape
        n_w13 = w13_e4m3.shape[1]
        n_inter = n_w13 // 2
        h = w2_e4m3.shape[1]
        total_slots = tokens * top_k
        num_m_blocks = expert_ids.numel()

        if act_buf is None:
            act_buf = torch.empty((total_slots, n_inter), dtype=torch.bfloat16, device=x.device)
        if f32_buf is None:
            f32_buf = torch.zeros((tokens, h), dtype=torch.float32, device=x.device)
        if out_buf is None:
            out_buf = torch.empty((tokens, h), dtype=torch.bfloat16, device=x.device)

        if block_m >= 64:
            bn, bk, nw, ns = 64, 64, 8, 3
        elif block_m >= 32:
            bn, bk, nw, ns = 64, 64, 4, 3
        else:
            bn, bk, nw, ns = 32, 128, 4, 3
        _mxfp8_moe_w13_situ_kernel[(num_m_blocks, triton.cdiv(n_w13, bn))](
            x,
            w13_e4m3.view(torch.uint8),
            w13_ue8m0,
            topk_weights.reshape(-1),
            sorted_ids,
            expert_ids,
            act_buf,
            total_slots,
            N=n_w13,
            N_INTER=n_inter,
            K=k,
            TOP_K=top_k,
            BM=block_m,
            BN=bn,
            BK=bk,
            num_warps=nw,
            num_stages=ns,
        )
        _mxfp8_moe_w2_kernel[(num_m_blocks, triton.cdiv(h, bn))](
            act_buf,
            w2_e4m3.view(torch.uint8),
            w2_ue8m0,
            sorted_ids,
            expert_ids,
            f32_buf,
            total_slots,
            N=h,
            K=n_inter,
            TOP_K=top_k,
            BM=block_m,
            BN=bn,
            BK=bk,
            num_warps=nw,
            num_stages=ns,
        )
        n_el = tokens * h
        _cast_and_zero_kernel[(triton.cdiv(n_el, 512),)](f32_buf, out_buf, n_el, BLOCK=512, num_warps=4)
        return out_buf

    return run_nvfp4_grouped_moe, run_mxfp8_grouped_moe


def main() -> None:
    torch.manual_seed(11)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_experts = 8
    top_k = 8
    print(f"=== Lab 3B: Batched Top-8 SM86 MoE (`NVFP4` + `MXFP8`) on 8 Real Experts ({device}) ===")

    # 1. Fetch 8 REAL Layer 0 NVFP4 experts (~35.8 MB) and 8 REAL Layer 20 MXFP8 experts (~65.4 MB)
    #    Both vastly exceed RTX 3090's 6 MB L2 cache -> 100% honest GDDR6X measurements!
    nv = fetch_real_nvfp4_expert_sample(layer_id=0, num_experts=num_experts)
    mx = fetch_real_mxfp8_expert_sample(layer_id=20, num_experts=num_experts)

    w13_nv_bf16 = dequant_nvfp4_weight(
        nv["w13_weight"], nv["w13_weight_scale"], nv["w13_weight_amax"], nv["w13_denom"]
    ).to(device)
    w2_nv_bf16 = dequant_nvfp4_weight(
        nv["w2_weight"], nv["w2_weight_scale"], nv["w2_weight_amax"], nv["w2_denom"]
    ).to(device)
    w13_mx_bf16 = dequant_mxfp8_weight(mx["w13_weight"], mx["w13_weight_scale"]).to(device)
    w2_mx_bf16 = dequant_mxfp8_weight(mx["w2_weight"], mx["w2_weight_scale"]).to(device)

    x_check = (torch.randn(4, 2304, device=device, dtype=torch.bfloat16) * 0.5).contiguous()
    ids_check, wts_check = make_synthetic_routing(4, num_experts, top_k, device)
    ref_nv = batched_moe_ref(x_check, w13_nv_bf16, w2_nv_bf16, ids_check, wts_check)
    ref_mx = batched_moe_ref(x_check, w13_mx_bf16, w2_mx_bf16, ids_check, wts_check)
    print(
        f"[Real 8-Expert Oracle] Layer 0 NVFP4 std={ref_nv.float().std().item():.5f} | "
        f"Layer 20 MXFP8 std={ref_mx.float().std().item():.5f}"
    )

    if not torch.cuda.is_available():
        s_ids, e_ids, n_pad = moe_align_block_size(ids_check, num_experts, block_m=16)
        assert int(n_pad.item()) == s_ids.numel() and e_ids.numel() * 16 == s_ids.numel()
        print("[CPU Mode] `moe_align_block_size` + 8-expert real NVFP4/MXFP8 golden references PASS!")
        return

    # Move quantized tensors to RTX 3090
    nv_w13_p = nv["w13_weight"].to(device).contiguous()
    nv_w13_s = nv["w13_weight_scale"].to(device).contiguous()
    nv_w13_o = (nv["w13_weight_amax"].float() / nv["w13_denom"].float()).to(device).contiguous()
    nv_w2_p = nv["w2_weight"].to(device).contiguous()
    nv_w2_s = nv["w2_weight_scale"].to(device).contiguous()
    nv_w2_o = (nv["w2_weight_amax"].float() / nv["w2_denom"].float()).to(device).contiguous()

    mx_w13_w = mx["w13_weight"].to(device).contiguous()
    mx_w13_s = mx["w13_weight_scale"].to(device).contiguous()
    mx_w2_w = mx["w2_weight"].to(device).contiguous()
    mx_w2_s = mx["w2_weight_scale"].to(device).contiguous()

    run_nvfp4_grouped, run_mxfp8_grouped = build_sm86_batched_moe_triton_kernels()
    cuda_moe = load_cuda_moe_extension()

    # 2. Strict numerical parity across ALL engines on real 8-expert Inkling-2-Mini weights (T=4)
    s_ids4, e_ids4, _ = moe_align_block_size(ids_check, num_experts, block_m=16)
    tri_nv = run_nvfp4_grouped(
        x_check, nv_w13_p, nv_w13_s, nv_w13_o, nv_w2_p, nv_w2_s, nv_w2_o,
        wts_check, s_ids4, e_ids4, top_k=top_k,
    )
    tri_mx = run_mxfp8_grouped(
        x_check, mx_w13_w, mx_w13_s, mx_w2_w, mx_w2_s, wts_check, s_ids4, e_ids4, top_k=top_k,
    )
    c1, c2 = run_nvfp4_grouped.last_compiled  # type: ignore[attr-defined]

    act4 = torch.empty((4, top_k, 1152), dtype=torch.bfloat16, device=device)
    out4 = torch.empty((4, 2304), dtype=torch.bfloat16, device=device)
    f324 = torch.zeros((4, 2304), dtype=torch.float32, device=device)

    cu_nv_fused = cuda_moe.nvfp4_moe_decode_forward(
        x_check, nv_w13_p, nv_w13_s.view(torch.uint8), nv_w13_o,
        nv_w2_p, nv_w2_s.view(torch.uint8), nv_w2_o,
        ids_check, wts_check, act4, out4.clone(), f324, False,
    )
    cu_nv_atomic = cuda_moe.nvfp4_moe_decode_forward(
        x_check, nv_w13_p, nv_w13_s.view(torch.uint8), nv_w13_o,
        nv_w2_p, nv_w2_s.view(torch.uint8), nv_w2_o,
        ids_check, wts_check, act4, out4.clone(), f324, True,
    )
    cu_mx_fused = cuda_moe.mxfp8_moe_decode_forward(
        x_check, mx_w13_w.view(torch.uint8), mx_w13_s,
        mx_w2_w.view(torch.uint8), mx_w2_s,
        ids_check, wts_check, act4, out4.clone(), f324,
    )

    err_tri_nv = (tri_nv.float() - ref_nv.float()).abs().max().item()
    err_cu_nv_f = (cu_nv_fused.float() - ref_nv.float()).abs().max().item()
    err_cu_nv_a = (cu_nv_atomic.float() - ref_nv.float()).abs().max().item()
    err_tri_mx = (tri_mx.float() - ref_mx.float()).abs().max().item()
    err_cu_mx_f = (cu_mx_fused.float() - ref_mx.float()).abs().max().item()

    print(f"\nReal 8-Expert Top-8 MoE Parity (T=4, Triton regs: w13+SiTU={c1.n_regs}, w2={c2.n_regs}):")
    print(f"  NVFP4 Triton Grouped lop3+SiTU max|err|      : {err_tri_nv:.5e}")
    print(f"  NVFP4 CUDA Warp-GEMV (fused w2) max|err|     : {err_cu_nv_f:.5e}")
    print(f"  NVFP4 CUDA Warp-GEMV (atomic w2) max|err|    : {err_cu_nv_a:.5e}")
    print(f"  MXFP8 Triton Grouped + SiTU max|err|         : {err_tri_mx:.5e}")
    print(f"  MXFP8 CUDA Warp-GEMV (fused w2) max|err|     : {err_cu_mx_f:.5e}")
    assert max(err_tri_nv, err_cu_nv_f, err_cu_nv_a) < 2e-2
    assert max(err_tri_mx, err_cu_mx_f) < 2e-2

    # 3. CUDA-Graph pure-GPU benchmark across T = 1, 4, 16, 64!
    def bench_graph(fn, iters: int = 80) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
        for _ in range(2):
            g.replay()
        torch.cuda.synchronize()
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_ev.record()
        g.replay()
        e_ev.record()
        torch.cuda.synchronize()
        return (s_ev.elapsed_time(e_ev) * 1000.0) / iters

    nv_bytes = (
        nv_w13_p.numel() + nv_w13_s.numel() + nv_w2_p.numel() + nv_w2_s.numel()
    )
    mx_bytes = (
        mx_w13_w.numel() + mx_w13_s.numel() + mx_w2_w.numel() + mx_w2_s.numel()
    )

    # Optional production Humming 0.1.10 SM86 (GemmType.INDEXED) baseline (already cached!)
    h_w13 = h_w2 = run_situ = None
    try:
        from humming import dtypes as h_dtypes
        from humming.config import WeightScaleType
        from humming.layer import HummingLayer
        from humming.schema import HummingWeightSchema
        from labs.lab3_sm86_nvfp4_mxfp8_moe import build_sm86_moe_triton_kernels

        _, _, run_situ = build_sm86_moe_triton_kernels()
        schema_nv = HummingWeightSchema(
            b_dtype=h_dtypes.float4e2m1,
            bs_dtype=h_dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP_TENSOR,
        )
        h_w13 = HummingLayer(2304, 2304, weight_config=schema_nv, num_experts=num_experts, torch_dtype=torch.bfloat16).to(device)
        h_w2 = HummingLayer(2304, 1152, weight_config=schema_nv, num_experts=num_experts, torch_dtype=torch.bfloat16).to(device)
        with torch.no_grad():
            h_w13.weight.copy_(nv_w13_p.view(h_w13.weight.dtype))
            h_w13.weight_scale.copy_(nv_w13_s.view(h_w13.weight_scale.dtype))
            h_w13.global_scale.copy_(nv_w13_o.reshape(h_w13.global_scale.shape))
            h_w2.weight.copy_(nv_w2_p.view(h_w2.weight.dtype))
            h_w2.weight_scale.copy_(nv_w2_s.view(h_w2.weight_scale.dtype))
            h_w2.global_scale.copy_(nv_w2_o.reshape(h_w2.global_scale.shape))
        h_w13.transform()
        h_w2.transform()
    except Exception:
        h_w13 = h_w2 = run_situ = None

    print(
        f"\n=== NVFP4 Top-8 MoE Crossover Benchmark (8 Real Experts = {nv_bytes / 1e6:.2f} MB; "
        f"GA102 SOL @ 840 GB/s = {nv_bytes / 840e3:.1f} us) ==="
    )
    print(
        f"  {'T':>3} | {'BM':>2} | {'Dense BF16':>11} | {'Humming IDX':>11} | {'Triton lop3':>11} | "
        f"{'CUDA GEMV(A)':>12} | {'Best Eff BW':>11} | {'Best vs BF16':>12}"
    )

    nv_t1_best = 0.0
    nv_t1_hum = 0.0
    for tokens in (1, 4, 16, 64):
        xt = (torch.randn(tokens, 2304, device=device, dtype=torch.bfloat16) * 0.5).contiguous()
        t_ids, t_wts = make_synthetic_routing(tokens, num_experts, top_k, device)
        bm = choose_moe_block_m(tokens, top_k, num_experts)
        s_ids, e_ids, _ = moe_align_block_size(t_ids, num_experts, block_m=bm)

        act_b = torch.empty((tokens * top_k, 1152), dtype=torch.bfloat16, device=device)
        act_3d = act_b.view(tokens, top_k, 1152)
        f32_b = torch.zeros((tokens, 2304), dtype=torch.float32, device=device)
        out_b = torch.empty((tokens, 2304), dtype=torch.bfloat16, device=device)

        def dense_bf16_moe():
            x_exp = xt.unsqueeze(0).expand(num_experts, tokens, 2304)
            gu = torch.bmm(x_exp, w13_nv_bf16.transpose(1, 2))  # [E, T, 2*I]
            g = gu[..., 0::2].float()
            u = gu[..., 1::2].float()
            act = (((4.0 * torch.tanh(g * 0.25)) * torch.sigmoid(g)) * (25.0 * torch.tanh(u * 0.04)))
            act = (act * t_wts.T.unsqueeze(-1)).to(torch.bfloat16)
            dn = torch.bmm(act, w2_nv_bf16.transpose(1, 2))
            return dn.sum(dim=0)

        t_bf16 = bench_graph(dense_bf16_moe, iters=50)
        t_tri = bench_graph(
            lambda: run_nvfp4_grouped(
                xt, nv_w13_p, nv_w13_s, nv_w13_o, nv_w2_p, nv_w2_s, nv_w2_o,
                t_wts, s_ids, e_ids, top_k=top_k, block_m=bm,
                act_buf=act_b, f32_buf=f32_b, out_buf=out_b,
            ),
            iters=60,
        )
        t_cu_a = bench_graph(
            lambda: cuda_moe.nvfp4_moe_decode_forward(
                xt, nv_w13_p, nv_w13_s.view(torch.uint8), nv_w13_o,
                nv_w2_p, nv_w2_s.view(torch.uint8), nv_w2_o,
                t_ids, t_wts, act_3d, out_b, f32_b, True,
            ),
            iters=60,
        )
        if h_w13 is not None and h_w2 is not None and run_situ is not None:
            hum_bm = 16 if tokens <= 16 else 64
            h_si, h_ei, h_npad = moe_align_block_size(t_ids, num_experts, block_m=hum_bm)
            h_gu = torch.empty((tokens * top_k, 2304), dtype=torch.bfloat16, device=device)
            h_dn = torch.empty((tokens * top_k, 2304), dtype=torch.bfloat16, device=device)

            def run_humming_moe():
                cfg = {"gemm_type": "indexed"}
                h_w13(xt, outputs=h_gu, sorted_ids=h_si, expert_ids=h_ei, num_tokens_padded=h_npad, top_k=top_k, compute_config=cfg)
                act = run_situ(h_gu, t_wts.reshape(-1))
                h_w2(act, outputs=h_dn, sorted_ids=h_si, expert_ids=h_ei, num_tokens_padded=h_npad, top_k=1, compute_config=cfg)
                return h_dn.view(tokens, top_k, 2304).sum(dim=1)

            t_hum = bench_graph(run_humming_moe, iters=50)
            hum_str = f"{t_hum:8.2f} us"
        else:
            t_hum = float("inf")
            hum_str = f"{'n/a':>11}"

        best = min(t_tri, t_cu_a)
        eff_bw = (nv_bytes / 1e9) / (best / 1e6)
        if tokens == 1:
            nv_t1_best = best
            nv_t1_hum = t_hum
        print(
            f"  {tokens:3d} | {bm:2d} | {t_bf16:8.2f} us | {hum_str} | {t_tri:8.2f} us | "
            f"{t_cu_a:9.2f} us | {eff_bw:6.1f} GB/s | {t_bf16 / best:11.2f}x"
        )

    print(
        f"\n=== MXFP8 Top-8 MoE Crossover Benchmark (8 Real Experts = {mx_bytes / 1e6:.2f} MB; "
        f"GA102 SOL @ 840 GB/s = {mx_bytes / 840e3:.1f} us) ==="
    )
    print(
        f"  {'T':>3} | {'BM':>2} | {'Dense BF16':>11} | {'Triton MXFP8':>13} | "
        f"{'CUDA GEMV(A)':>13} | {'Best Eff BW':>11} | {'Best vs BF16':>12}"
    )
    mx_t1_best = 0.0
    for tokens in (1, 4, 16, 64):
        xt = (torch.randn(tokens, 2304, device=device, dtype=torch.bfloat16) * 0.5).contiguous()
        t_ids, t_wts = make_synthetic_routing(tokens, num_experts, top_k, device)
        bm = choose_moe_block_m(tokens, top_k, num_experts)
        s_ids, e_ids, _ = moe_align_block_size(t_ids, num_experts, block_m=bm)

        act_b = torch.empty((tokens * top_k, 1152), dtype=torch.bfloat16, device=device)
        act_3d = act_b.view(tokens, top_k, 1152)
        f32_b = torch.zeros((tokens, 2304), dtype=torch.float32, device=device)
        out_b = torch.empty((tokens, 2304), dtype=torch.bfloat16, device=device)

        def dense_bf16_mx():
            x_exp = xt.unsqueeze(0).expand(num_experts, tokens, 2304)
            gu = torch.bmm(x_exp, w13_mx_bf16.transpose(1, 2))
            g = gu[..., 0::2].float()
            u = gu[..., 1::2].float()
            act = (((4.0 * torch.tanh(g * 0.25)) * torch.sigmoid(g)) * (25.0 * torch.tanh(u * 0.04)))
            act = (act * t_wts.T.unsqueeze(-1)).to(torch.bfloat16)
            return torch.bmm(act, w2_mx_bf16.transpose(1, 2)).sum(dim=0)

        t_bf16 = bench_graph(dense_bf16_mx, iters=40)
        t_tri = bench_graph(
            lambda: run_mxfp8_grouped(
                xt, mx_w13_w, mx_w13_s, mx_w2_w, mx_w2_s, t_wts, s_ids, e_ids, top_k=top_k, block_m=bm,
                act_buf=act_b, f32_buf=f32_b, out_buf=out_b,
            ),
            iters=50,
        )
        t_cu_a = bench_graph(
            lambda: cuda_moe.mxfp8_moe_decode_forward(
                xt, mx_w13_w.view(torch.uint8), mx_w13_s,
                mx_w2_w.view(torch.uint8), mx_w2_s,
                t_ids, t_wts, act_3d, out_b, f32_b,
            ),
            iters=50,
        )
        best = min(t_tri, t_cu_a)
        eff_bw = (mx_bytes / 1e9) / (best / 1e6)
        if tokens == 1:
            mx_t1_best = best
        print(
            f"  {tokens:3d} | {bm:2d} | {t_bf16:8.2f} us | {t_tri:10.2f} us | "
            f"{t_cu_a:10.2f} us | {eff_bw:6.1f} GB/s | {t_bf16 / best:11.2f}x"
        )

    total_48_ms = (44 * nv_t1_best + 4 * mx_t1_best) / 1000.0
    sol_48_ms = (44 * (nv_bytes / 840e3) + 4 * (mx_bytes / 840e3)) / 1000.0
    print(
        f"\n[Full Inkling-2-Mini 48-Visit MoE Decode Projection @ T=1 on 1x RTX 3090]:\n"
        f"  44x NVFP4 ({nv_t1_best:.2f} us) + 4x MXFP8 ({mx_t1_best:.2f} us) = {total_48_ms:.2f} ms/tok "
        f"({1000.0 / total_48_ms:.1f} tok/s MoE ceiling; SOL={sol_48_ms:.2f} ms = {sol_48_ms / total_48_ms * 100:.1f}% of 840 GB/s!)"
    )
    if nv_t1_hum < float("inf"):
        print(
            f"  vs Humming SM86 INDEXED on 44x NVFP4: {44 * nv_t1_best / 1000.0:.2f} ms vs "
            f"{44 * nv_t1_hum / 1000.0:.2f} ms ({nv_t1_hum / nv_t1_best:.2f}x faster at T=1 decode!)"
        )


if __name__ == "__main__":
    main()
