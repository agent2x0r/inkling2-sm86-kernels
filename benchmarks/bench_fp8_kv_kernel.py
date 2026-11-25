#!/usr/bin/env python3
"""Standalone SM86 (RTX 3090) microbenchmark & bit-exactness verifier for software FP8 E4M3FN KV.

Verifies without downloading the model checkpoint:
  1. Software FP32 -> E4M3FN (`uint8`) RNE store path: 200,125/200,125 bit-exact vs `torch.float8_e4m3fn`
  2. Software E4M3FN -> FP16x2 5-op SM86 PTX loader: 254/254 finite values bit-exact & 0.00e+00 attention diff
  3. L2-flushed SM86 paged-attention throughput vs BF16 (Decode 16K..128K, DSpark Verify 16K..128K, Prefill 32K)

Run on an Ampere GPU (inside `vllm-inkling2-sm86:latest` or a CUDA PyTorch + Triton env):
    python3 benchmarks/bench_fp8_kv_kernel.py --out benchmarks/results_kernel_fp8kv.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32_to_e4m3fn_u8(x):
    """Exact IEEE-754 RNE fp32 -> saturating E4M3FN byte (`uint8`) for `< SM89`."""
    u = x.to(tl.uint32, bitcast=True)
    sign = (u >> 24) & 0x80
    abs_u = u & 0x7FFFFFFF
    clamped = tl.minimum(abs_u, 0x43E00000)
    norm_mag = ((clamped + 0x0007FFFF + ((clamped >> 20) & 1)) - 0x3C000000) >> 20
    sub_mag = (
        (abs_u.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True)
        - 0x46800000
    )
    mag = tl.where(abs_u < 0x3C800000, sub_mag, norm_mag)
    mag = tl.where(abs_u > 0x7F800000, 0x7F, mag)
    return (sign | mag).to(tl.uint8)


@triton.jit
def _k_store_sw(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(Y + offs, _fp32_to_e4m3fn_u8(tl.load(X + offs, mask=mask, other=0.0)), mask=mask)


@triton.jit
def _load_fp8_kv_sm86(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    """5-op SM80/SM86 PTX unpack (`prmt`+`and`+`shr`+`lop3`+non-FTZ `mul.rn.f16x2` * 256.0)."""
    half = tl.arange(0, D // 2)
    pair = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, D // 2)[:, None]
        + half[None, :],
        mask=mask[:, None],
        other=0,
    )
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 p32, s, em, h2, sc; "
        "cvt.u32.u16 p32, $2; "
        "prmt.b32 s, p32, 0, 0x1707; "
        "and.b32 em, s, 0x7F007F00; "
        "shr.u32 em, em, 1; "
        "lop3.b32 h2, s, em, 0x80008000, 0xEC; "
        "mov.b32 sc, 0x5C005C00; "
        "mul.rn.f16x2 h2, h2, sc; "
        "mov.b32 {$0, $1}, h2; }",
        "=h,=h,h",
        [pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.interleave(lo, hi).to(dtype)


@triton.jit
def _k_load_test(X, Y, D: tl.constexpr):
    offs = tl.arange(0, 2) * D
    out = _load_fp8_kv_sm86(X, offs, offs < 256, tl.bfloat16, D)
    tl.store(Y + offs[:, None] + tl.arange(0, D)[None, :], out)


@triton.jit
def _attn_bench_kernel(
    Q, K, V, BlockTable, SeqLens, CuSeqlensQ, Out, Lse,
    softmax_scale, num_tokens, num_splits,
    stride_q_t, stride_q_h, stride_kv_blk, stride_kv_slot, stride_kv_h,
    H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, FP8: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
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
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    n_hi = (tl.minimum(pos_hi + 1, seqlen_k) + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - split + num_splits - 1, 0) // num_splits
    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None], other=0.0,
    )
    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64)
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        if FP8:
            k = _load_fp8_kv_sm86(K, kv_ptrs, key_valid, q.dtype, D).T
        else:
            k = tl.load(K + kv_ptrs[None, :] + offs_d[:, None], mask=key_valid[None, :], other=0.0).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        s = tl.where((dist >= 0) & key_valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if FP8:
            v = _load_fp8_kv_sm86(V, kv_ptrs, key_valid, q.dtype, D)
        else:
            v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    part = (split * num_tokens + token) * H + head
    tl.store(Lse + part, lse2, mask=row_valid)
    tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


def _bench_l2_flushed(fn, flush_buf, warmup: int = 10, iters: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        flush_buf.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 1. Store exactness
    specials = torch.tensor(
        [
            0.0, -0.0, 1e-40, -1e-40, 2**-11, 2**-10, -2**-10,
            *[k * 2**-10 for k in range(32)],
            *[-k * 2**-10 for k in range(32)],
            *[k * 2**-9 * (1.0 + 1e-6) for k in range(16)],
            *[k * 2**-9 * (1.0 - 1e-6) for k in range(16)],
            0.015625, 1.0, 1.0625, 1.125, 1.1875, 1.25, -1.0625,
            22.627417, -22.627417, 400.0, 416.0, 432.0, 447.9, 448.0, 448.1, 464.0, 500.0, -500.0,
            float("inf"), float("-inf"), float("nan"), -float("nan"),
        ],
        device="cuda",
        dtype=torch.float32,
    )
    torch.manual_seed(0)
    xs = torch.cat([
        specials,
        torch.randn(50_000, device="cuda") * 0.01,
        torch.randn(50_000, device="cuda") * 1.0,
        torch.randn(50_000, device="cuda") * 25.0,
        torch.randn(50_000, device="cuda") * 300.0,
    ])
    y_sw = torch.empty(xs.numel(), device="cuda", dtype=torch.uint8)
    _k_store_sw[(triton.cdiv(xs.numel(), 1024),)](xs, y_sw, xs.numel(), BLOCK=1024, enable_fp_fusion=False)
    y_ref = xs.to(torch.float8_e4m3fn).view(torch.uint8)
    store_exact = int((y_sw == y_ref).sum().item())
    assert store_exact == xs.numel(), f"Store mismatch: {store_exact}/{xs.numel()}"
    print(f"[PASS] Software FP32->E4M3FN store bit-exact: {store_exact:,}/{xs.numel():,}")

    # 2. Load exactness
    all_u8 = torch.arange(256, device="cuda", dtype=torch.uint8)
    out_bf16 = torch.empty(256, device="cuda", dtype=torch.bfloat16)
    _k_load_test[(1,)](all_u8, out_bf16, D=128)
    ref_bf16 = all_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
    finite = (all_u8 != 0x7F) & (all_u8 != 0xFF)
    assert (out_bf16.view(torch.uint16) == ref_bf16.view(torch.uint16))[finite].all().item()
    print("[PASS] Software E4M3FN->FP16x2->BF16 loader bit-exact across all 254 finite values")

    # 3. L2-flushed SM86 attention throughput (TP=2 shape: H=12, HKV=4, D=128)
    H, HKV, D, PAGE = 12, 4, 128, 16
    G = H // HKV
    flush = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.int32)
    configs = [
        ("Decode q=1, 16K KV", 1, 16384, 20, 16, 128, 4),
        ("Decode q=1, 64K KV", 1, 65536, 20, 16, 128, 4),
        ("Decode q=1, 128K KV", 1, 131072, 20, 16, 128, 4),
        ("DSpark Verify q=8, 16K KV", 8, 16384, 20, 32, 128, 4),
        ("DSpark Verify q=8, 64K KV", 8, 65536, 20, 32, 128, 4),
        ("DSpark Verify q=8, 128K KV", 8, 131072, 20, 32, 128, 4),
        ("Prefill q=256, 32K KV", 256, 32768, 1, 128, 64, 8),
    ]
    rows = []
    print(f"\n{'Regime':28s} | {'BF16 (us)':>10s} | {'FP8 (us)':>10s} | {'Speedup':>8s} | {'FP8 GB/s':>9s} | {'Max Diff':>9s}")
    print("-" * 88)
    for label, q_len, seqlen_k, num_splits, bm, bn, warps in configs:
        num_blocks = seqlen_k // PAGE
        q = torch.randn(q_len, H, D, device="cuda", dtype=torch.bfloat16) * 0.5
        k_bf16 = torch.randn(num_blocks, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16) * 0.5
        v_bf16 = torch.randn(num_blocks, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16) * 0.5
        k_fp8 = k_bf16.float().to(torch.float8_e4m3fn)
        v_fp8 = v_bf16.float().to(torch.float8_e4m3fn)
        k_u8, v_u8 = k_fp8.view(torch.uint8), v_fp8.view(torch.uint8)
        k_dq, v_dq = k_fp8.to(torch.bfloat16), v_fp8.to(torch.bfloat16)
        bt = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)
        seqlens = torch.tensor([seqlen_k], device="cuda", dtype=torch.int32)
        cu_q = torch.tensor([0, q_len], device="cuda", dtype=torch.int32)
        o_bf = torch.empty((num_splits, q_len, H, D), device="cuda", dtype=torch.float32)
        o_fp = torch.empty_like(o_bf)
        lse = torch.empty((num_splits, q_len, H), device="cuda", dtype=torch.float32)
        grid = (triton.cdiv(q_len, bm // G), HKV * num_splits, 1)

        fn_bf = lambda: _attn_bench_kernel[grid](
            q, k_dq, v_dq, bt, seqlens, cu_q, o_bf, lse,
            D**-0.5, q_len, num_splits,
            q.stride(0), q.stride(1), k_dq.stride(0), k_dq.stride(1), k_dq.stride(2),
            H=H, G=G, D=D, PAGE=PAGE, BLOCK_M=bm, BLOCK_N=bn, FP8=False,
            num_warps=warps, num_stages=2,
        )
        fn_fp = lambda: _attn_bench_kernel[grid](
            q, k_u8, v_u8, bt, seqlens, cu_q, o_fp, lse,
            D**-0.5, q_len, num_splits,
            q.stride(0), q.stride(1), k_u8.stride(0), k_u8.stride(1), k_u8.stride(2),
            H=H, G=G, D=D, PAGE=PAGE, BLOCK_M=bm, BLOCK_N=bn, FP8=True,
            num_warps=warps, num_stages=2,
        )
        us_bf = _bench_l2_flushed(fn_bf, flush)
        us_fp = _bench_l2_flushed(fn_fp, flush)
        diff = (o_bf - o_fp).abs().max().item()
        bw = (seqlen_k * HKV * D * 2) / (us_fp * 1e-6) / 1e9
        rows.append({
            "regime": label,
            "q_len": q_len,
            "seqlen_k": seqlen_k,
            "splits": num_splits,
            "bf16_us": round(us_bf, 1),
            "fp8_us": round(us_fp, 1),
            "speedup": round(us_bf / us_fp, 2),
            "fp8_read_gbs": round(bw, 1),
            "max_diff_vs_dequant_bf16": diff,
        })
        print(f"{label:28s} | {us_bf:10.1f} | {us_fp:10.1f} | {us_bf/us_fp:7.2f}x | {bw:9.1f} | {diff:9.1e}")

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "store_exact_matches": f"{store_exact}/{xs.numel()}",
                    "load_finite_exact": "254/254",
                    "regimes": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
