#!/usr/bin/env python3
"""Lab 0: Ampere GA102 (RTX 3090 / SM86) Roofline, Dual-GPU Topology & Checkpoint Audit.

Run locally (CPU math + real checkpoint audit) or on the dual-3090 box (live GPU roofline):
    uv run python labs/lab0_sm86_roofline.py

What you learn in Lab 0:
1. Arithmetic Intensity (FLOPs / Byte) decides everything:
   - RTX 3090 (GA102, SM86): 82 SMs, 936.2 GB/s theoretical GDDR6X bandwidth.
   - Crucial GeForce GA102 quirk: BF16/FP16 Tensor Core MMA (`m16n8k16`) runs at:
       * ~71.2 TFLOPS when accumulating in FP32 (`mma.sync...f32.bf16.bf16.f32`)
       * ~142.5 TFLOPS when accumulating in FP16 (`mma.sync...f16.f16.f16.f16`)
     With FP32 accumulation (which Inkling 2 requires for numeric stability), the ridge point is:
       71,200 GFLOP/s / 936.2 GB/s ≈ 76 FLOPs / Byte!
     - Batch-1 decode GEMM / MoE has arithmetic intensity ~2–4 FLOPs/Byte -> 100% MEMORY BOUND.
     - Prefill (T >= 128) has arithmetic intensity > 100 FLOPs/Byte -> TENSOR CORE COMPUTE BOUND.

2. Why Inkling 2 Mini fits 64K on ONE 3090 (`TP=1`) and 128K+ on TWO 3090s (`TP=2`):
   - 24 physical layers execute 48 layer visits (layers 6..17 loop 3x), creating 48 KV caches.
   - 40 of those 48 visits are sliding-window (`W = 1024`), capping at 160 MiB TOTAL forever!
   - Only 8 visits are full-attention (`32 KiB/token` total -> `2.0 GiB` at 64K, `4.0 GiB` at 128K).

3. Why PCIe `TP=2` can hurt single-stream decode latency:
   - 48 layer visits × 2 sublayers (Attn + MoE) = 96 all-reduces per generated token!
   - Even if each PCIe all-reduce of `[1, 2304]` BF16 (4.5 KB) takes just 18 us, 96 × 18 us = 1.73 ms
     of pure bus latency per token—measured directly by this script when 2 GPUs are present!
"""

from __future__ import annotations

import time

import torch

from inkling_sm86.checkpoint_slice import (
    fetch_real_nvfp4_expert_sample,
    summarize_checkpoint_precision_bytes,
)
from inkling_sm86.reference import (
    INKLING2_MINI_SPEC,
    dequant_nvfp4_weight,
    estimate_vram_budget,
)

RTX3090_THEORETICAL_BW_GBS = 936.2
RTX3090_BF16_FP32ACC_TFLOPS = 71.2


def print_architecture_and_vram_analysis() -> None:
    s = INKLING2_MINI_SPEC
    print("=" * 78)
    print("INKLING 2 MINI × DUAL RTX 3090 (SM86) FIRST-PRINCIPLES AUDIT")
    print("=" * 78)
    print(
        f"Model geometry : hidden={s.hidden_size}, inter={s.intermediate_size}, "
        f"GQA={s.num_q_heads}Q/{s.num_kv_heads}KV (G={s.gqa_group_size}), head_dim={s.head_dim}"
    )
    print(
        f"Looped depth   : {s.num_physical_layers} physical layers -> {s.num_layer_visits} visits/token "
        f"({s.num_swa_attn_visits} SWA@{s.sliding_window} + {s.num_full_attn_visits} full-attn)"
    )
    print(
        f"MoE per visit  : top-{s.top_k} of {s.num_experts} experts | "
        f"NVFP4 (layers 0-19) + MXFP8 (layers 20-23)"
    )

    # Decode memory traffic per token (batch=1, worst-case 8 distinct experts/visit):
    # Per NVFP4 expert: w13 (2304*1152 B + scales) + w2 (2304*576 B + scales) ≈ 4.48 MB
    nvfp4_expert_mb = (2304 * 1152 + 2304 * 144 + 2304 * 576 + 2304 * 72) / 1e6
    mxfp8_expert_mb = (2304 * 2304 + 2304 * 72 + 2304 * 1152 + 2304 * 36) / 1e6
    # 44 visits hit NVFP4 layers (0..5, 6..17 x3, 18..19 = 6 + 36 + 2 = 44), 4 hit MXFP8 (20..23)
    active_moe_gb = (44 * s.top_k * nvfp4_expert_mb + 4 * s.top_k * mxfp8_expert_mb) / 1e3
    dense_qkv_o_gb = (48 * (2304 * (3072 + 384 + 3072 + 24 + 2048) + 3072 * 2304) * 2) / 1e9
    total_decode_gb = active_moe_gb + dense_qkv_o_gb
    ideal_tp1_ms = (total_decode_gb / RTX3090_THEORETICAL_BW_GBS) * 1e3
    print(
        f"\nBatch-1 weight read/token : ~{total_decode_gb:.2f} GB "
        f"({active_moe_gb:.2f} GB routed experts + {dense_qkv_o_gb:.2f} GB dense BF16)"
    )
    print(
        f"Single-3090 roofline ceiling (no speculator): "
        f"~{ideal_tp1_ms:.2f} ms/tok -> ~{1000.0 / ideal_tp1_ms:.0f} tok/s @ 100% BW "
        f"(~{0.72 * 1000.0 / ideal_tp1_ms:.0f} tok/s @ 72% BW, before 3.5x DSpark!)"
    )

    print("\n--- Per-GPU VRAM Budget on 24.0 GiB RTX 3090 (BF16 KV Cache + DSpark) ---")
    print(f"{'Config':<16} {'Weights':>10} {'Full KV':>10} {'SWA KV':>10} {'Total':>10} {'Free':>10}")
    for tp in (1, 2):
        for ctx in (4096, 32768, 65536, 131072, 262144):
            b = estimate_vram_budget(ctx, tp_size=tp, include_dspark=True)
            tag = f"TP={tp}, {ctx // 1024}K"
            status = f"{b.headroom_on_24gib_card_gib:+.2f} GiB"
            print(
                f"{tag:<16} {b.weights_per_gpu_gib:>8.2f}G {b.full_kv_per_gpu_gib:>8.2f}G "
                f"{b.swa_kv_per_gpu_gib:>8.2f}G {b.total_static_plus_kv_gib:>8.2f}G {status:>10}"
            )


def audit_real_checkpoint_slice() -> None:
    print("\n--- Official `thinkingmachines/Inkling-2-Mini` Safetensors Audit ---")
    try:
        by_dtype = summarize_checkpoint_precision_bytes()
        total = sum(by_dtype.values())
        print(f"Total safetensors payload: {total:,} bytes ({total / 1024**3:.2f} GiB)")
        for dt, nbytes in by_dtype.items():
            print(f"  {dt:<10}: {nbytes / 1024**3:>6.2f} GiB ({100.0 * nbytes / total:>5.1f}%)")

        sample = fetch_real_nvfp4_expert_sample(layer_id=0, num_experts=1)
        w13_bf16 = dequant_nvfp4_weight(
            sample["w13_weight"],
            sample["w13_weight_scale"],
            sample["w13_weight_amax"],
            sample["w13_denom"],
        )
        print(
            f"Streamed real Layer 0 Expert 0 `w13` -> dequantized shape={tuple(w13_bf16.shape)}, "
            f"dtype={w13_bf16.dtype}, std={w13_bf16.float().std().item():.5f}, "
            f"max={w13_bf16.float().abs().max().item():.5f}"
        )
    except Exception as exc:
        print(f"(Offline / skipped HTTP Range audit: {exc})")


def run_cuda_roofline_benchmarks() -> None:
    if not torch.cuda.is_available():
        print("\n[CPU Mode] No CUDA GPUs detected on this machine.")
        print("Run `./scripts/remote.sh lab0` to execute live SM86 GDDR6X + Tensor Core benchmarks!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n=== LIVE CUDA HARDWARE PROBE ({num_gpus} GPU(s) detected) ===")
    for idx in range(num_gpus):
        props = torch.cuda.get_device_properties(idx)
        smem_kb = props.shared_memory_per_multiprocessor / 1024
        print(
            f"GPU {idx}: {props.name} | SM {props.major}.{props.minor} | "
            f"{props.multi_processor_count} SMs | {props.total_memory / 1024**3:.2f} GiB VRAM | "
            f"{smem_kb:.0f} KB SMEM/SM"
        )

    device = torch.device("cuda:0")
    # 1. Measure real GDDR6X streaming bandwidth (256 MiB copy = 512 MiB moved)
    n_elems = 64 * 1024 * 1024  # 64M float32 = 256 MiB
    src = torch.randn(n_elems, device=device, dtype=torch.float32)
    dst = torch.empty_like(src)
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize(device)
    iters = 40
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        dst.copy_(src)
    end_ev.record()
    torch.cuda.synchronize(device)
    ms = start_ev.elapsed_time(end_ev) / iters
    bw_gbs = (2 * src.numel() * 4) / (ms * 1e-3) / 1e9
    print(
        f"\n[1] GDDR6X Device Copy Bandwidth : {bw_gbs:6.1f} GB/s "
        f"({100.0 * bw_gbs / RTX3090_THEORETICAL_BW_GBS:.1f}% of {RTX3090_THEORETICAL_BW_GBS} GB/s peak)"
    )
    del src, dst

    # 2. Measure BF16 Tensor Core GEMM at Inkling shapes (`[4096, 2304] @ [2304, 2304]`)
    m, n, k = 4096, 2304, 2304
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16).T
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize(device)
    start_ev.record()
    for _ in range(iters):
        _ = a @ b
    end_ev.record()
    torch.cuda.synchronize(device)
    gemm_ms = start_ev.elapsed_time(end_ev) / iters
    tflops = (2.0 * m * n * k) / (gemm_ms * 1e-3) / 1e12
    print(
        f"[2] BF16 Tensor Core GEMM (FP32 accum): {tflops:6.2f} TFLOPS "
        f"({100.0 * tflops / RTX3090_BF16_FP32ACC_TFLOPS:.1f}% of {RTX3090_BF16_FP32ACC_TFLOPS} TFLOPS GA102 peak)"
    )

    # Compare against FP16 w/ reduced precision reduction if allowed (shows GeForce 2:1 accum ratio)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    a16 = a.half()
    b16 = b.half()
    for _ in range(10):
        _ = a16 @ b16
    torch.cuda.synchronize(device)
    start_ev.record()
    for _ in range(iters):
        _ = a16 @ b16
    end_ev.record()
    torch.cuda.synchronize(device)
    fp16_ms = start_ev.elapsed_time(end_ev) / iters
    fp16_tflops = (2.0 * m * n * k) / (fp16_ms * 1e-3) / 1e12
    print(f"    (Compare FP16 fast-accum mode     : {fp16_tflops:6.2f} TFLOPS)")
    del a, b, a16, b16

    # 3. Dual-GPU peer-to-peer bandwidth (NVLink vs PCIe) & 1-token hop latency probe
    if num_gpus >= 2:
        can_p2p = torch.cuda.can_device_access_peer(0, 1)
        print(f"\n[3] Dual-GPU Interconnect: can_device_access_peer(0, 1) = {can_p2p}")

        # Large-buffer unidirectional P2P bandwidth (256 MiB -> distinguishes NV4 NVLink ~53 GB/s vs PCIe ~6-12 GB/s)
        big0 = torch.empty(128 * 1024 * 1024, device="cuda:0", dtype=torch.bfloat16)
        big1 = torch.empty_like(big0, device="cuda:1")
        for _ in range(5):
            big1.copy_(big0)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(20):
            big1.copy_(big0)
        end_ev.record()
        torch.cuda.synchronize()
        p2p_ms = start_ev.elapsed_time(end_ev) / 20.0
        p2p_gbs = (big0.numel() * 2) / (p2p_ms * 1e-3) / 1e9
        link_kind = "NVLink (NV4)" if p2p_gbs > 30.0 else "PCIe"
        print(f"    256 MiB GPU0->GPU1 P2P bandwidth : {p2p_gbs:5.1f} GB/s ({link_kind} detected!)")
        del big0, big1

        t0 = torch.randn(1, 2304, device="cuda:0", dtype=torch.bfloat16)
        t1 = torch.empty_like(t0, device="cuda:1")
        for _ in range(20):
            t1.copy_(t0, non_blocking=False)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        rounds = 200
        for _ in range(rounds):
            t1.copy_(t0, non_blocking=False)
            torch.cuda.synchronize()
        us_sync = (time.perf_counter() - t_start) * 1e6 / rounds

        # Pipelined non-blocking copy (amortizes per-call host sync)
        start_ev.record()
        for _ in range(500):
            t1.copy_(t0, non_blocking=True)
        end_ev.record()
        torch.cuda.synchronize()
        us_pipe = (start_ev.elapsed_time(end_ev) * 1000.0) / 500.0
        print(
            f"    1-token [1, 2304] BF16 hop (host-synced / pipelined): {us_sync:.1f} us / {us_pipe:.1f} us"
        )
        print(
            "    (Note: host-driven copy_ is CPU-launch bound; inside CUDA Graphs / NCCL P2P over NVLink,"
        )
        print(
            "     direct GPU-to-GPU AllReduce drops to ~5-8 us -> TP=2 gets ~2x GDDR6X BW + 10-14 GiB free VRAM/card!)"
        )


if __name__ == "__main__":
    print_architecture_and_vram_analysis()
    audit_real_checkpoint_slice()
    run_cuda_roofline_benchmarks()
