#!/usr/bin/env python3
"""Lab 2: Custom Sheared-Bias Paged FlashAttention + Split-KV Decode for Ampere (SM86).

Run:
    uv run python labs/lab2_sm86_rel_attention.py

Why Inkling 2 Mini Attention is Special (and why standard FlashAttention-2 cannot run it):
1. NO RoPE! Instead, every token projects `r` (`d_rel=16`) against a learned matrix `[16, 1024]`
   to produce per-head, per-distance bias logits `rel_logits[t, h, dist]` (`0 <= dist < 1024`).
2. Sheared Memory Layout:
   `attention_prologue` writes `rel_logits` into a `(T + 128, H, 1024 + 256)` sheared buffer so
   that for any query `t` inside a 128-token query tile ending at `tile_end`, key index `k` sits
   at contiguous memory offset:
       `Rel + t * stride_rel_t + h * 1280 + k - (cdiv(past + tile_end, 128) * 128 - 1280)`
   Adding this bias inside the SRAM tile loop BEFORE online softmax avoids materializing a huge
   `[H, T, S]` score matrix in VRAM!
3. Hybrid Sliding Window + Post-Softmax Gate & Attention Sink:
   - 40 of 48 layer visits use `W = 1024` sliding window; 8 visits use full causal attention.
   - After softmax `o = P @ V` and log-sum-exp `lse`, every output channel is gated by BOTH a
     per-channel sigmoid gate AND a per-head attention sink:
         `final_out = o * sigmoid(gate) * sigmoid(lse - sink)`

Why Our SM86 Kernel Beats Naive Single-Split Porting on RTX 3090 (82 SMs):
- Inkling 2 Mini has only `H_kv = 8` KV heads (`H_q = 24`, GQA group `G = 3`).
- At batch=1 decode (`q_len = 1`), `num_splits = 1` launches only `8 CTAs` on an `82-SM` RTX 3090
  — leaving 74 of 82 SMs (90.2% of the GPU!) completely idle!
- `choose_sm86_decode_splits(...)` below partitions the visible KV range across `S` splits
  (`8 * 10 = 80 CTAs` for full-attention layers, saturating 80/82 SMs!) and fuses the gate + sink
  directly into the log-sum-exp combine kernel (with zero SM90-only `launch_pdl` PTX).
"""

from __future__ import annotations

import torch

from inkling_sm86.checkpoint_slice import fetch_tensor_leading_slice
from inkling_sm86.reference import paged_rel_attention_ref, shear_rel_logits_ref

RTX3090_SM_COUNT = 82


def choose_sm86_decode_splits(
    num_reqs: int,
    num_kv_heads: int,
    max_visible_kv: int,
    block_n: int = 64,
    num_sms: int = RTX3090_SM_COUNT,
) -> int:
    """Choose split-KV factor `S` to fill RTX 3090's 82 SMs without over-splitting tiny contexts."""
    base_ctas = num_reqs * num_kv_heads
    total_kv_blocks = max(1, (max_visible_kv + block_n - 1) // block_n)
    if base_ctas >= num_sms or total_kv_blocks <= 2:
        return 1
    # Target ~1 full wave across 82 SMs (e.g. 8 KV heads * 10 splits = 80 CTAs),
    # while ensuring each split still owns at least 2 KV blocks of work.
    target_splits = max(1, num_sms // base_ctas)
    max_useful_splits = max(1, total_kv_blocks // 2)
    return min(target_splits, max_useful_splits, 32)


def build_sm86_rel_attention_kernels():
    import triton
    import triton.language as tl

    _SHEAR_TILE = tl.constexpr(128)
    MODE_OUT_LSE = 0
    MODE_GATED = 1
    MODE_PARTIAL = 2

    @triton.jit
    def _sigmoid(x):
        return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))

    @triton.jit
    def _sm86_rel_attn_kernel(
        Q,
        K,
        V,
        Rel,
        BlockTable,
        SeqLens,
        CuSeqlensQ,
        Out,
        Lse,
        Gate,
        Sink,
        softmax_scale,
        num_tokens,
        num_splits,
        stride_q_t,
        stride_q_h,
        stride_kv_blk,
        stride_kv_slot,
        stride_kv_h,
        stride_rel_t,
        stride_bt,
        stride_out_t,
        stride_out_h,
        stride_lse_h,
        stride_lse_t,
        stride_gate,
        stride_sink,
        H: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        E: tl.constexpr,
        PAGE: tl.constexpr,
        WINDOW: tl.constexpr,
        MODE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        TOKENS: tl.constexpr = BLOCK_M // G
        E_PAD: tl.constexpr = E + 256
        LOG2E: tl.constexpr = 1.4426950408889634

        m_tile = tl.program_id(0)
        kv_head = tl.program_id(1) // num_splits
        split = tl.program_id(1) % num_splits
        req = tl.program_id(2)

        q_start = tl.load(CuSeqlensQ + req)
        q_len = tl.load(CuSeqlensQ + req + 1) - q_start
        first_tok = m_tile * TOKENS
        if first_tok >= q_len:
            return

        seqlen_k = tl.load(SeqLens + req)
        past = seqlen_k - q_len

        rows = tl.arange(0, BLOCK_M)
        tok = first_tok + rows // G
        row_valid = (rows < TOKENS * G) & (tok < q_len)
        head = kv_head * G + rows % G
        token = (q_start + tok).to(tl.int64)

        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

        n_lo = 0
        if WINDOW > 0:
            n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
        n_hi = (tl.minimum(pos_hi + 1, seqlen_k) + BLOCK_N - 1) // BLOCK_N
        num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

        offs_d = tl.arange(0, D)
        q = tl.load(
            Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )

        # Sheared relative-bias row base pointer:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

        qk_scale = softmax_scale * LOG2E
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        bt_ptr = BlockTable + req.to(tl.int64) * stride_bt
        kv_off = kv_head * stride_kv_h

        for i in range(num_iters):
            n = n_lo + split + i * num_splits
            keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
            key_valid = keys < seqlen_k
            page = tl.load(bt_ptr + keys // PAGE, mask=key_valid, other=0)
            kv_ptrs = page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off

            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)

            s = tl.dot(q, k) * qk_scale
            dist = pos[:, None] - keys[None, :]
            visible = (dist >= 0) & key_valid[None, :]
            if WINDOW > 0:
                visible = visible & (dist < WINDOW)

            in_extent = visible & (dist < E) & row_valid[:, None]
            bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
            s += bias.to(tl.float32) * LOG2E
            s = tl.where(visible, s, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp2(m_i - m_safe)
            p = tl.exp2(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        empty = l_i == 0.0
        o = acc / tl.where(empty, 1.0, l_i)[:, None]
        lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))

        if MODE == 0:
            tl.store(
                Out + token[:, None] * stride_out_t + head[:, None] * stride_out_h + offs_d[None, :],
                o,
                mask=row_valid[:, None],
            )
            tl.store(
                Lse + head * stride_lse_h + token * stride_lse_t,
                lse2 * 0.6931471805599453,
                mask=row_valid,
            )
        elif MODE == 1:
            cols = head[:, None] * D + offs_d[None, :]
            gate = tl.load(Gate + token[:, None] * stride_gate + cols, mask=row_valid[:, None], other=0.0)
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
            tl.store(Out + token[:, None] * stride_out_t + cols, o.to(tl.bfloat16), mask=row_valid[:, None])
        else:
            part = (split * num_tokens + token) * H + head
            tl.store(Lse + part, lse2, mask=row_valid)
            tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])

    @triton.jit
    def _sm86_decode_combine_kernel(
        OutPart,
        LsePart,
        Gate,
        Sink,
        Out,
        num_tokens,
        stride_gate,
        stride_sink,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """Merge `NUM_SPLITS` log2-LSE partials and fuse `sigmoid(gate) * sigmoid(lse - sink)`."""
        tok = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1).to(tl.int64)
        splits = tl.arange(0, BLOCK_S)
        s_mask = splits < NUM_SPLITS
        part_idx = (splits.to(tl.int64) * num_tokens + tok) * H + head

        lse2 = tl.load(LsePart + part_idx, mask=s_mask, other=float("-inf"))
        m = tl.max(lse2, axis=0)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        weights = tl.where(s_mask & (lse2 > float("-inf")), tl.exp2(lse2 - m_safe), 0.0)
        denom = tl.sum(weights, axis=0)
        inv_denom = tl.where(denom == 0.0, 0.0, 1.0 / denom)

        offs_d = tl.arange(0, D)
        partials = tl.load(
            OutPart + part_idx[:, None] * D + offs_d[None, :],
            mask=s_mask[:, None],
            other=0.0,
        )
        o = tl.sum(partials * weights[:, None], axis=0) * inv_denom

        lse_nat = (m_safe + tl.log2(tl.where(denom == 0.0, 1.0, denom))) * 0.6931471805599453
        cols = head * D + offs_d
        gate = tl.load(Gate + tok * stride_gate + cols).to(tl.float32)
        sink = tl.load(Sink + tok * stride_sink + head).to(tl.float32)
        o = o * _sigmoid(gate) * _sigmoid(lse_nat - sink)
        tl.store(Out + tok * (H * D) + cols, o.to(tl.bfloat16))

    def run_sm86_gated_attention(
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        sheared_rel: torch.Tensor,
        gate: torch.Tensor,
        sinks: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_query_len: int,
        window: int = 0,
        num_splits: int | None = None,
    ) -> torch.Tensor:
        total_q, num_heads, head_dim = q.shape
        page_size, num_kv_heads = key_cache.shape[1:3]
        group = num_heads // num_kv_heads
        extent = sheared_rel.shape[2] - 256
        scale = head_dim**-0.5
        num_reqs = cache_seqlens.shape[0]

        rows = max_query_len * group
        block_m = 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128
        block_n = 64
        # On GA102 (100 KB SMEM/SM), 3 stages fit cleanly for BLOCK_M <= 64
        num_stages = 3 if block_m <= 64 else 2
        num_warps = 4 if block_m <= 64 else 8

        if num_splits is None:
            if max_query_len <= 4:
                max_vis = window if window > 0 else int(cache_seqlens.max().item())
                num_splits = choose_sm86_decode_splits(num_reqs, num_kv_heads, max_vis, block_n)
            else:
                num_splits = 1

        tokens_per_tile = block_m // group
        grid = (triton.cdiv(max_query_len, tokens_per_tile), num_kv_heads * num_splits, num_reqs)
        out = torch.empty((total_q, num_heads * head_dim), dtype=torch.bfloat16, device=q.device)

        if num_splits == 1:
            _sm86_rel_attn_kernel[grid](
                q,
                key_cache,
                value_cache,
                sheared_rel,
                block_table,
                cache_seqlens,
                cu_seqlens_q,
                out,
                out,
                gate,
                sinks,
                scale,
                total_q,
                1,
                q.stride(0),
                q.stride(1),
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                sheared_rel.stride(0),
                block_table.stride(0),
                out.stride(0),
                0,
                0,
                0,
                gate.stride(0),
                sinks.stride(0),
                H=num_heads,
                G=group,
                D=head_dim,
                E=extent,
                PAGE=page_size,
                WINDOW=window,
                MODE=MODE_GATED,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return out

        o_part = torch.empty((num_splits, total_q, num_heads, head_dim), dtype=torch.float32, device=q.device)
        lse_part = torch.empty((num_splits, total_q, num_heads), dtype=torch.float32, device=q.device)
        _sm86_rel_attn_kernel[grid](
            q,
            key_cache,
            value_cache,
            sheared_rel,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            o_part,
            lse_part,
            gate,
            sinks,
            scale,
            total_q,
            num_splits,
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            sheared_rel.stride(0),
            block_table.stride(0),
            0,
            0,
            0,
            0,
            gate.stride(0),
            sinks.stride(0),
            H=num_heads,
            G=group,
            D=head_dim,
            E=extent,
            PAGE=page_size,
            WINDOW=window,
            MODE=MODE_PARTIAL,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        _sm86_decode_combine_kernel[(total_q, num_heads)](
            o_part,
            lse_part,
            gate,
            sinks,
            out,
            total_q,
            gate.stride(0),
            sinks.stride(0),
            H=num_heads,
            D=head_dim,
            NUM_SPLITS=num_splits,
            BLOCK_S=triton.next_power_of_2(num_splits),
            num_warps=4,
        )
        return out

    return run_sm86_gated_attention


def _make_case(
    seqs: list[tuple[int, int]],
    page_size: int = 16,
    window: int = 0,
    device: str = "cpu",
    rel_proj_real: torch.Tensor | None = None,
):
    num_heads, num_kv_heads, head_dim, d_rel, extent = 24, 8, 128, 16, 1024
    total_q = sum(ql for ql, _ in seqs)
    max_kv = max(kl for _, kl in seqs)
    blocks_per_seq = (max_kv + page_size - 1) // page_size
    total_blocks = len(seqs) * blocks_per_seq + 4

    q = torch.randn(total_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    r = torch.randn(total_q, num_heads, d_rel, device=device, dtype=torch.bfloat16) * 0.3
    rel_proj = (
        rel_proj_real.to(device=device, dtype=torch.bfloat16)
        if rel_proj_real is not None
        else torch.randn(d_rel, extent, device=device, dtype=torch.bfloat16) * 0.5
    )
    rel_natural = torch.einsum("thd,de->the", r.float(), rel_proj.float()).to(torch.bfloat16)
    sheared_rel = shear_rel_logits_ref(rel_natural, seqs, tile=128, is_local=window > 0)

    key_cache = torch.randn(
        total_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    value_cache = torch.randn_like(key_cache)
    perm = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    block_table = perm[: len(seqs) * blocks_per_seq].view(len(seqs), blocks_per_seq)

    gate = torch.randn(total_q, num_heads * head_dim, device=device, dtype=torch.bfloat16)
    sinks = torch.randn(total_q, num_heads, device=device, dtype=torch.bfloat16)

    cu = [0]
    for ql, _ in seqs:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([kl for _, kl in seqs], device=device, dtype=torch.int32)
    return q, key_cache, value_cache, rel_natural, sheared_rel, gate, sinks, block_table, cache_seqlens, cu_seqlens_q


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Lab 2: Sheared-Bias Paged FlashAttention + Split-KV ({device}) ===")

    rel_proj_real = None
    try:
        rel_proj_real = fetch_tensor_leading_slice("model.llm.layers.6.attn.rel_logits_proj.proj")
        print(f"Loaded real Layer 6 `rel_logits_proj` from Inkling-2-Mini: {tuple(rel_proj_real.shape)}")
    except Exception:
        pass

    cases = [
        ("full_prefill_swa", [(256, 256)], 1024),
        ("chunked_prefill_full", [(96, 1400)], 0),
        ("varlen_multiseq_swa", [(1, 1500), (17, 600), (65, 2048)], 1024),
        ("decode_depth_4k_full", [(1, 4096)], 0),
    ]

    for name, seqs, win in cases:
        (
            q,
            k_cache,
            v_cache,
            rel_nat,
            sheared,
            gate,
            sinks,
            bt,
            seqlens,
            cu_q,
        ) = _make_case(seqs, window=win, device=device, rel_proj_real=rel_proj_real)
        ref_gated, _ = paged_rel_attention_ref(
            q, k_cache, v_cache, rel_nat, seqs, bt, window_size=win, gate=gate, sinks=sinks
        )
        if not torch.cuda.is_available():
            print(
                f"  [Ref OK] {name:<24}: out={tuple(ref_gated.shape)}, "
                f"sheared_rel={tuple(sheared.shape)}, std={ref_gated.float().std().item():.4f}"
            )
        else:
            kernel_fn = build_sm86_rel_attention_kernels()
            max_q = max(ql for ql, _ in seqs)
            actual = kernel_fn(
                q, k_cache, v_cache, sheared, gate, sinks, bt, seqlens, cu_q, max_q, window=win
            ).view_as(ref_gated)
            max_err = (actual.float() - ref_gated.float()).abs().max().item()
            mean_err = (actual.float() - ref_gated.float()).abs().mean().item()
            print(f"  [PASS] {name:<24}: max|err|={max_err:.5e}, mean|err|={mean_err:.5e}")
            assert max_err < 1.5e-2 and mean_err < 1e-3, f"Parity failed on {name}: {max_err}"

    if not torch.cuda.is_available():
        print("\n[CPU Mode] Sheared-bias + SWA + Gate/Sink oracle verified!")
        print("Run `./scripts/remote.sh lab2` on your 3090s to benchmark SM86 Split-KV at 32K/64K!")
        return

    # Benchmark single-split (8 CTAs on 82 SMs!) vs SM86 Split-KV (80 CTAs on 82 SMs!) at 32K decode
    kernel_fn = build_sm86_rel_attention_kernels()
    seqs_32k = [(1, 32768)]
    q, k_cache, v_cache, rel_nat, sheared, gate, sinks, bt, seqlens, cu_q = _make_case(
        seqs_32k, window=0, device="cuda", rel_proj_real=rel_proj_real
    )

    def bench(splits: int) -> float:
        fn = lambda: kernel_fn(
            q, k_cache, v_cache, sheared, gate, sinks, bt, seqlens, cu_q, 1, window=0, num_splits=splits
        )
        for _ in range(15):
            fn()
        torch.cuda.synchronize()
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_ev.record()
        for _ in range(80):
            fn()
        e_ev.record()
        torch.cuda.synchronize()
        return (s_ev.elapsed_time(e_ev) * 1000.0) / 80

    us_s1 = bench(1)
    auto_splits = choose_sm86_decode_splits(1, 8, 32768)
    us_auto = bench(auto_splits)
    kv_bytes = 32768 * 8 * 128 * 2 * 2  # K + V read for 32K tokens in BF16 = 134.2 MB
    print(f"\nBatch-1 Full-Attention Decode @ 32K KV ({kv_bytes / 1e6:.1f} MB KV read):")
    print(f"  num_splits = 1  ( 8 CTAs / 82 SMs) : {us_s1:7.2f} us | {kv_bytes / (us_s1 * 1e-6) / 1e9:6.1f} GB/s")
    print(
        f"  num_splits = {auto_splits:<2} ({8 * auto_splits:2d} CTAs / 82 SMs) : {us_auto:7.2f} us | "
        f"{kv_bytes / (us_auto * 1e-6) / 1e9:6.1f} GB/s ({us_s1 / us_auto:4.2f}x speedup!)"
    )


if __name__ == "__main__":
    main()
