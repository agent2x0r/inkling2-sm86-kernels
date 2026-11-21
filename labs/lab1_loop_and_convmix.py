#!/usr/bin/env python3
"""Lab 1: Memory-Bound Fusion — Triton vs. Handwritten SM86 CUDA C++ (`cp.async` + Warp Shuffles).

Run:
    uv run python labs/lab1_loop_and_convmix.py

What you build and compare in Lab 1:
1. Inkling 2 Mini's looped core (layers 6..17) runs 3 passes. In passes 2 and 3, each sublayer
   (Q, K, V, MoE -> `S = 4` streams) merges a cross-pass `message` into `dest`:
       x = (1 - tanh(forget)) * dest + tanh(gain) * message     [FP32, NO FMA contraction!]
       y = RMSNorm(x, norm_weight, eps=1e-6)                    [rounded once to BF16]

2. Why fusion matters on an RTX 3090 (936 GB/s GDDR6X):
   - Unfused PyTorch launches 7 separate elementwise/reduction kernels, reading and writing
     intermediate `(S=4, T, D=2304)` FP32 tensors to GDDR6X ~6 times!
   - Our fused Triton kernel AND our handwritten CUDA C++ kernel (`csrc/convmix_sm86.cu`) read
     the inputs ONCE, keep `x` in registers during the RMSNorm reduction, and write BF16 ONCE.

3. Triton -> CUDA C++ side-by-side:
   - Read `triton_loop_blend_rmsnorm` right below (35 lines of Triton), then open
     `csrc/convmix_sm86.cu` to see the exact same register layout, `__fmul_rn` non-FMA
     semantics, `__shfl_xor_sync` warp butterfly sum, and Ampere `cp.async.ca.shared.global`
     16-byte DMA staging for causal ShortConv (`k=4`)!
"""

from __future__ import annotations

from pathlib import Path

import torch

from inkling_sm86.reference import loop_blend_rms_norm_ref

CSRC_PATH = Path(__file__).resolve().parent.parent / "csrc" / "convmix_sm86.cu"


def short_conv_rmsnorm_ref(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    residual: torch.Tensor,
    norm_w: torch.Tensor,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden PyTorch reference for static causal depthwise ShortConv (k=4) + residual + RMSNorm."""
    t_len, d = x.shape
    padded = torch.nn.functional.pad(x.float().T.unsqueeze(0), (3, 0))  # [1, D, T + 3]
    weight = conv_w.float().unsqueeze(1)  # [D, 1, 4]
    conv = torch.nn.functional.conv1d(padded, weight, groups=d).squeeze(0).T * (5.0**-0.5)
    conv_out = (residual.float() + conv).to(torch.bfloat16)
    xf = conv_out.float()
    rstd = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + norm_eps)
    normed = (xf * rstd * norm_w.float()).to(torch.bfloat16)
    assert normed.shape == (t_len, d)
    return conv_out, normed


def build_triton_loop_blend():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _triton_blend_rmsnorm_kernel(
        Dest,
        Message,
        GainForget,
        NormW,
        Out,
        stride_dest_s,
        stride_w_s,
        T,
        norm_eps,
        D: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        s = tl.program_id(0).to(tl.int64)
        t = tl.program_id(1).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        row = s * T + t

        dest = tl.load(Dest + s * stride_dest_s + t * D + cols, mask=mask, other=0.0)
        msg = tl.load(Message + row * D + cols, mask=mask, other=0.0)
        gain = libdevice.tanh(tl.load(GainForget + row * (2 * D) + cols, mask=mask, other=0.0))
        forget = libdevice.tanh(
            tl.load(GainForget + row * (2 * D) + D + cols, mask=mask, other=0.0)
        )

        # Kept in registers across the reduction! Compiled with enable_fp_fusion=False.
        x = (1.0 - forget) * dest + gain * msg
        mean_sq = tl.sum(x * x, axis=0) / D
        rstd = tl.rsqrt(mean_sq + norm_eps)
        w = tl.load(NormW + s * stride_w_s + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd) * w
        tl.store(Out + row * D + cols, y.to(tl.bfloat16), mask=mask)

    def launch(
        dest: torch.Tensor,
        message: torch.Tensor,
        gain_forget: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        s, t, d = message.shape
        out = torch.empty_like(message, dtype=torch.bfloat16)
        _triton_blend_rmsnorm_kernel[(s, t)](
            dest,
            message,
            gain_forget,
            norm_weight,
            out,
            dest.stride(0),
            norm_weight.stride(0),
            t,
            norm_eps,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            num_warps=8,
            enable_fp_fusion=False,
        )
        return out

    return launch


def load_sm86_cuda_extension():
    from torch.utils.cpp_extension import load

    return load(
        name="inkling_convmix_sm86_ext",
        sources=[str(CSRC_PATH)],
        extra_cuda_cflags=["-O3", "--fmad=false", "-lineinfo"],
        verbose=False,
    )


def _bench_cuda_us(fn, warmup: int = 15, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(iters):
        fn()
    e_ev.record()
    torch.cuda.synchronize()
    return (s_ev.elapsed_time(e_ev) * 1000.0) / iters


def _bench_graph_us(fn, warmup: int = 10, graph_iters: int = 200) -> float:
    """Capture `graph_iters` launches in a CUDA Graph to strip FX-8350 host launch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(graph_iters):
            fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    s_ev.record()
    g.replay()
    e_ev.record()
    torch.cuda.synchronize()
    return (s_ev.elapsed_time(e_ev) * 1000.0) / graph_iters


def main() -> None:
    torch.manual_seed(0)
    streams, tokens, hidden = 4, 256, 2304
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Lab 1: Looped Gated Blend + ShortConv/RMSNorm ({device}) ===")

    dest = torch.randn(1, tokens, hidden, device=device, dtype=torch.float32).expand(
        streams, -1, -1
    )
    message = torch.randn(streams, tokens, hidden, device=device, dtype=torch.float32)
    gain_forget = torch.randn(streams, tokens, 2 * hidden, device=device, dtype=torch.float32)
    norm_weight = torch.randn(streams, hidden, device=device, dtype=torch.bfloat16)

    ref_out = loop_blend_rms_norm_ref(dest, message, gain_forget, norm_weight)
    print(
        f"PyTorch Golden Reference computed: shape={tuple(ref_out.shape)}, dtype={ref_out.dtype}"
    )

    x_conv = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w_conv = torch.randn(hidden, 4, device=device, dtype=torch.bfloat16)
    res_conv = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    nw_conv = torch.randn(hidden, device=device, dtype=torch.bfloat16)
    ref_conv_out, ref_conv_norm = short_conv_rmsnorm_ref(x_conv, w_conv, res_conv, nw_conv)
    print(
        f"ShortConv+RMSNorm Reference computed: shape={tuple(ref_conv_norm.shape)}, "
        f"std={ref_conv_norm.float().std().item():.4f}"
    )

    if not torch.cuda.is_available():
        print("\n[CPU Mode] References verified. Run `./scripts/remote.sh lab1` on your 3090s")
        print("to JIT-compile both the Triton kernel and `csrc/convmix_sm86.cu` and benchmark GB/s!")
        return

    # 1. Verify & benchmark Triton kernel
    triton_fn = build_triton_loop_blend()
    tri_out = triton_fn(dest, message, gain_forget, norm_weight)
    tri_err = (tri_out.float() - ref_out.float()).abs().max().item()

    # 2. Compile, verify & benchmark handwritten SM86 CUDA C++ kernel (`csrc/convmix_sm86.cu`)
    print("Compiling `csrc/convmix_sm86.cu` with `nvcc -O3 --fmad=false`...")
    cuda_ext = load_sm86_cuda_extension()
    cu_out = cuda_ext.loop_blend_rmsnorm_sm86(dest, message, gain_forget, norm_weight, 1e-6)
    cu_err = (cu_out.float() - ref_out.float()).abs().max().item()

    cu_conv_out, cu_conv_norm = cuda_ext.short_conv_rmsnorm_sm86(
        x_conv, w_conv, res_conv, nw_conv, 1e-6
    )
    cu_conv_out_v2, cu_conv_norm_v2 = cuda_ext.short_conv_rmsnorm_sm86_v2(
        x_conv, w_conv, res_conv, nw_conv, 1e-6
    )
    # Also verify non-multiple-of-8 tail handling (T = 13)!
    ref_tail_c, ref_tail_n = short_conv_rmsnorm_ref(x_conv[:13], w_conv, res_conv[:13], nw_conv)
    v2_tail_c, v2_tail_n = cuda_ext.short_conv_rmsnorm_sm86_v2(
        x_conv[:13].contiguous(), w_conv, res_conv[:13].contiguous(), nw_conv, 1e-6
    )
    torch.testing.assert_close(v2_tail_c, ref_tail_c, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v2_tail_n, ref_tail_n, atol=1e-2, rtol=1e-2)

    conv_err = (cu_conv_norm.float() - ref_conv_norm.float()).abs().max().item()
    conv_state_err = (cu_conv_out.float() - ref_conv_out.float()).abs().max().item()
    conv_v2_err = (cu_conv_norm_v2.float() - ref_conv_norm.float()).abs().max().item()
    conv_v2_state_err = (cu_conv_out_v2.float() - ref_conv_out.float()).abs().max().item()

    tri_diff = (tri_out != ref_out).sum().item()
    cu_diff = (cu_out != ref_out).sum().item()
    conv_diff = (cu_conv_norm != ref_conv_norm).sum().item()
    conv_v2_diff = (cu_conv_norm_v2 != ref_conv_norm).sum().item()
    print(f"\nParity vs Golden Reference (S={streams}, T={tokens}, D={hidden}, N={ref_out.numel():,}):")
    print(f"  Triton Loop Blend + RMSNorm max |err|   : {tri_err:.6e} ({tri_diff} / {ref_out.numel():,} 1-ULP flips)")
    print(f"  CUDA C++ Loop Blend + RMSNorm max |err| : {cu_err:.6e} ({cu_diff} / {ref_out.numel():,} 1-ULP flips)")
    print(
        f"  CUDA ShortConv v1 (scalar, 1 tok/CTA)   : {conv_err:.6e} "
        f"({conv_diff} / {ref_conv_norm.numel():,} 1-ULP flips; conv_out: {conv_state_err:.6e})"
    )
    print(
        f"  CUDA ShortConv v2 (uint4, 8 tok ring)   : {conv_v2_err:.6e} "
        f"({conv_v2_diff} / {ref_conv_norm.numel():,} 1-ULP flips; conv_out: {conv_v2_state_err:.6e})"
    )
    torch.testing.assert_close(tri_out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cu_out, ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cu_conv_norm, ref_conv_norm, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cu_conv_out, ref_conv_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cu_conv_norm_v2, ref_conv_norm, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cu_conv_out_v2, ref_conv_out, atol=1e-2, rtol=1e-2)

    # Effective unique bytes moved for Loop Blend + RMSNorm:
    # Read dest [1, T, D] fp32 + message [S, T, D] fp32 + gain_forget [S, T, 2D] fp32 + norm_w [S, D] bf16
    # Write out [S, T, D] bf16
    bytes_blend = (
        tokens * hidden * 4
        + streams * tokens * hidden * 4
        + streams * tokens * 2 * hidden * 4
        + streams * hidden * 2
        + streams * tokens * hidden * 2
    )
    pt_us = _bench_cuda_us(lambda: loop_blend_rms_norm_ref(dest, message, gain_forget, norm_weight))
    tri_us = _bench_graph_us(lambda: triton_fn(dest, message, gain_forget, norm_weight))
    cu_us = _bench_graph_us(
        lambda: cuda_ext.loop_blend_rmsnorm_sm86(dest, message, gain_forget, norm_weight, 1e-6)
    )

    print("\nLoop Blend + RMSNorm (CUDA-Graph GPU time vs 840 GB/s measured GDDR6X ceiling):")
    for label, us in [
        ("Unfused PyTorch eager", pt_us),
        ("Fused Triton (`enable_fp_fusion=False`)", tri_us),
        ("Handwritten SM86 CUDA C++ (`float4` + warp shfl)", cu_us),
    ]:
        gbs = bytes_blend / (us * 1e-6) / 1e9
        print(f"  {label:<52}: {us:7.2f} us | {gbs:6.1f} GB/s ({pt_us / us:5.2f}x)")

    print("\nShortConv (k=4) + Residual + RMSNorm — Exercise 1A v1 vs v2 (CUDA-Graph GPU time):")
    for t_bench in (256, 2048):
        xb = torch.randn(t_bench, hidden, device=device, dtype=torch.bfloat16)
        rb = torch.randn(t_bench, hidden, device=device, dtype=torch.bfloat16)
        bytes_conv = (t_bench * hidden * 2) * 4 + hidden * 4 * 2 + hidden * 2
        pt_c = _bench_cuda_us(lambda: short_conv_rmsnorm_ref(xb, w_conv, rb, nw_conv))
        v1_c = _bench_graph_us(lambda: cuda_ext.short_conv_rmsnorm_sm86(xb, w_conv, rb, nw_conv, 1e-6))
        v2_c = _bench_graph_us(
            lambda: cuda_ext.short_conv_rmsnorm_sm86_v2(xb, w_conv, rb, nw_conv, 1e-6)
        )
        print(f"  [T={t_bench:4d}, unique traffic = {bytes_conv / 1e6:5.2f} MB]")
        print(f"    Unfused PyTorch conv1d + RMSNorm                  : {pt_c:7.2f} us | {bytes_conv / (pt_c * 1e-6) / 1e9:6.1f} GB/s ( 1.00x)")
        print(f"    v1 CUDA (`cp.async`, scalar bf16, 1 tok/CTA)      : {v1_c:7.2f} us | {bytes_conv / (v1_c * 1e-6) / 1e9:6.1f} GB/s ({pt_c / v1_c:5.2f}x)")
        print(
            f"    v2 CUDA (`uint4` + 288T + 8-tok `wait_group 1`)   : {v2_c:7.2f} us | "
            f"{bytes_conv / (v2_c * 1e-6) / 1e9:6.1f} GB/s ({pt_c / v2_c:5.2f}x, {v1_c / v2_c:.2f}x over v1!)"
        )


if __name__ == "__main__":
    main()
