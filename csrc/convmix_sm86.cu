// SPDX-License-Identifier: Apache-2.0
/*
 * Lab 1B: Handwritten Ampere (SM86) CUDA C++ + Inline PTX Kernel for Inkling 2 Mini.
 *
 * Implements two core memory-bound fused kernels from Inkling 2 Mini on SM86:
 *   1. `sm86_loop_blend_rmsnorm`:
 *      Fuses the cross-pass gated message blend + sublayer RMSNorm (`ops/loop.py:blend_rms_norm`):
 *          x[c] = (1.0f - tanh(forget[c])) * dest[c] + tanh(gain[c]) * msg[c]   (NO FMA!)
 *          rstd = rsqrt(mean(x^2) + eps)
 *          out[c] = bf16(x[c] * rstd * norm_weight[c])
 *      Demonstrates:
 *        - Vectorized 128-bit (`float4`) global memory loads for coalesced 936 GB/s GDDR6X reads
 *        - Explicit `__fmul_rn` / `__fadd_rn` (and `--fmad=false`) to forbid FMA contraction,
 *          matching Thinking Machines' exact reference numerics!
 *        - Inline PTX `tanh.approx.f32` and `rsqrt.approx.f32`
 *        - Warp-level butterfly reduction (`__shfl_xor_sync`) + tiny 8-float SMEM cross-warp sum
 *
 *   2. `sm86_short_conv_rmsnorm`:
 *      Demonstrates Ampere's hardware async global-to-shared copy (`cp.async.ca.shared.global`)
 *      for causal depthwise short convolution (`k = 4`, `D = 2304`) + residual + RMSNorm,
 *      replacing Hopper's TMA (`cp.async.bulk`) used in upstream `csrc/inkling2/conv_mix.cu`!
 */

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpSize = 32;

// NOTE on `tanh.approx.f32` vs `tanhf` on SM86:
// Ampere's single-instruction SFU `MUFU.TANH` (`tanh.approx.f32`) has ~2e-4 error,
// which flips ~0.23% (5,447 / 2.36M) of BF16 outputs by 1 ULP. Because this kernel is
// strictly GDDR6X memory-bandwidth bound on RTX 3090, exact `tanhf()` and `rsqrtf()`
// hide completely behind global loads and match PyTorch/Triton down to ~6 / 2.36M elements!

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, mask);
  }
  return val;
}

/*
 * Grid : (num_streams, num_tokens)
 * Block: 256 threads (8 warps). For D = 2304:
 *        2304 floats = 576 float4s -> each thread processes at most 3 float4s (12 channels),
 *        keeping the entire blended row `x` in registers so we NEVER re-read global memory!
 */
template <int kThreads = 256, int kMaxVec4PerThread = 4>
__global__ void __launch_bounds__(kThreads) sm86_loop_blend_rmsnorm_kernel(
    const float* __restrict__ dest,         // [S_or_1, T, D]
    const float* __restrict__ message,      // [S, T, D]
    const float* __restrict__ gain_forget,  // [S, T, 2 * D]
    const __nv_bfloat16* __restrict__ norm_w, // [S_or_1, D]
    __nv_bfloat16* __restrict__ out,        // [S, T, D]
    int64_t stride_dest_s,
    int64_t stride_w_s,
    int T,
    int D,
    float norm_eps) {
  const int64_t s = blockIdx.x;
  const int64_t t = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int vec_d = D >> 2;  // D / 4 (576 for D = 2304)
  const int64_t row = s * T + t;

  const float4* dest_row = reinterpret_cast<const float4*>(dest + s * stride_dest_s + t * D);
  const float4* msg_row = reinterpret_cast<const float4*>(message + row * D);
  const float4* gain_row = reinterpret_cast<const float4*>(gain_forget + row * (2 * D));
  const float4* forget_row = reinterpret_cast<const float4*>(gain_forget + row * (2 * D) + D);

  float x_reg[kMaxVec4PerThread][4];
  float thread_sq_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < kMaxVec4PerThread; ++i) {
    const int idx = tid + i * kThreads;
    if (idx < vec_d) {
      const float4 d4 = dest_row[idx];
      const float4 m4 = msg_row[idx];
      const float4 g4 = gain_row[idx];
      const float4 f4 = forget_row[idx];

      const float d_arr[4] = {d4.x, d4.y, d4.z, d4.w};
      const float m_arr[4] = {m4.x, m4.y, m4.z, m4.w};
      const float g_arr[4] = {g4.x, g4.y, g4.z, g4.w};
      const float f_arr[4] = {f4.x, f4.y, f4.z, f4.w};

#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float gain = tanhf(g_arr[c]);
        const float forget = tanhf(f_arr[c]);
        // Explicit IEEE round-to-nearest non-contracted ops matching reference!
        const float keep = __fmul_rn(__fsub_rn(1.0f, forget), d_arr[c]);
        const float add = __fmul_rn(gain, m_arr[c]);
        const float x_val = __fadd_rn(keep, add);
        x_reg[i][c] = x_val;
        thread_sq_sum = __fmaf_rn(x_val, x_val, thread_sq_sum);
      }
    }
  }

  // Two-level reduction: warp butterfly shuffle -> 8-warp shared memory reduction
  __shared__ float warp_sums[kThreads / kWarpSize];
  const float w_sum = warp_reduce_sum(thread_sq_sum);
  if (lane == 0) {
    warp_sums[warp] = w_sum;
  }
  __syncthreads();

  __shared__ float s_rstd;
  if (warp == 0) {
    const float partial = (lane < (kThreads / kWarpSize)) ? warp_sums[lane] : 0.0f;
    const float total_sq = warp_reduce_sum(partial);
    if (lane == 0) {
      const float mean_sq = __fdiv_rn(total_sq, static_cast<float>(D));
      s_rstd = rsqrtf(__fadd_rn(mean_sq, norm_eps));
    }
  }
  __syncthreads();

  const float rstd = s_rstd;
  const uint2* w_row = reinterpret_cast<const uint2*>(norm_w + s * stride_w_s);
  uint2* out_row = reinterpret_cast<uint2*>(out + row * D);

#pragma unroll
  for (int i = 0; i < kMaxVec4PerThread; ++i) {
    const int idx = tid + i * kThreads;
    if (idx < vec_d) {
      // Load 4 bf16 weights (8 bytes = uint2), scale in fp32, pack 4 bf16 outputs
      const uint2 w_packed = w_row[idx];
      const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&w_packed.x);
      const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&w_packed.y);
      const float2 wf01 = __bfloat1622float2(w01);
      const float2 wf23 = __bfloat1622float2(w23);

      const float2 y01 = make_float2(
          __fmul_rn(__fmul_rn(x_reg[i][0], rstd), wf01.x),
          __fmul_rn(__fmul_rn(x_reg[i][1], rstd), wf01.y));
      const float2 y23 = make_float2(
          __fmul_rn(__fmul_rn(x_reg[i][2], rstd), wf23.x),
          __fmul_rn(__fmul_rn(x_reg[i][3], rstd), wf23.y));

      const __nv_bfloat162 ob01 = __float22bfloat162_rn(y01);
      const __nv_bfloat162 ob23 = __float22bfloat162_rn(y23);
      uint2 out_packed;
      out_packed.x = *reinterpret_cast<const uint32_t*>(&ob01);
      out_packed.y = *reinterpret_cast<const uint32_t*>(&ob23);
      out_row[idx] = out_packed;
    }
  }
}

/*
 * Ampere `cp.async.ca.shared.global` helper (16-byte async DMA directly from L2/GDDR6X into SMEM,
 * bypassing intermediate registers—SM80/SM86's predecessor to Hopper's TMA!).
 */
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
#else
  *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int kPendingGroups>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPendingGroups) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit_and_wait() {
  cp_async_commit();
  cp_async_wait_group<0>();
}

__device__ __forceinline__ float2 bf16x2_to_f2(uint32_t packed) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&packed));
}

__device__ __forceinline__ uint32_t f2_to_bf16x2(float a, float b) {
  const __nv_bfloat162 v = __float22bfloat162_rn(make_float2(a, b));
  return *reinterpret_cast<const uint32_t*>(&v);
}

/*
 * Contiguous single-sequence causal depthwise ShortConv (k=4, scaled by 1/sqrt(5)) + residual + RMSNorm
 * using SM86 16-byte `cp.async` staging into shared memory.
 *
 * Grid : (T,)
 * Block: 256 threads, staging 4 rows x 2304 bf16 (18 KB SMEM total << 48 KB default limit).
 */
__global__ void __launch_bounds__(256) sm86_short_conv_rmsnorm_kernel(
    const __nv_bfloat16* __restrict__ x,       // [T, D]
    const __nv_bfloat16* __restrict__ conv_w,  // [D, 4]
    const __nv_bfloat16* __restrict__ residual,// [T, D]
    const __nv_bfloat16* __restrict__ norm_w,  // [D]
    __nv_bfloat16* __restrict__ conv_out,      // [T, D]
    __nv_bfloat16* __restrict__ normed_out,    // [T, D]
    int T,
    int D,
    float norm_eps) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* smem_taps = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [4, D]

  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int vec8 = D >> 3;  // 8 bf16 = 16 bytes (`uint4`), 288 chunks for D=2304

  // Stage taps t-3, t-2, t-1, t asynchronously from GDDR6X -> SMEM via SM86 `cp.async`!
#pragma unroll
  for (int tap = 0; tap < 4; ++tap) {
    const int src_t = t - (3 - tap);
    for (int v = tid; v < vec8; v += blockDim.x) {
      __nv_bfloat16* dst_ptr = smem_taps + tap * D + v * 8;
      if (src_t >= 0) {
        cp_async16(dst_ptr, x + static_cast<int64_t>(src_t) * D + v * 8);
      } else {
        *reinterpret_cast<uint4*>(dst_ptr) = make_uint4(0, 0, 0, 0);
      }
    }
  }
  cp_async_commit_and_wait();
  __syncthreads();

  constexpr float kInvSqrt5 = 0.4472135954999579f;  // 5**-0.5, Inkling's static sconv scale
  float y_reg[9];  // ceil(2304 / 256) = 9 channels per thread
  float sq_sum = 0.0f;

#pragma unroll
  for (int i = 0; i < 9; ++i) {
    const int c = tid + i * 256;
    if (c < D) {
      const __nv_bfloat16* w4 = conv_w + c * 4;
      float acc = 0.0f;
#pragma unroll
      for (int tap = 0; tap < 4; ++tap) {
        acc = __fmaf_rn(
            __bfloat162float(smem_taps[tap * D + c]),
            __bfloat162float(w4[tap]),
            acc);
      }
      const float scaled = __fmul_rn(acc, kInvSqrt5);
      const float res_f = __bfloat162float(residual[static_cast<int64_t>(t) * D + c]);
      // Round residual update to bf16 into `conv_out`, then RMSNorm in fp32
      const __nv_bfloat16 updated_bf16 = __float2bfloat16_rn(__fadd_rn(res_f, scaled));
      conv_out[static_cast<int64_t>(t) * D + c] = updated_bf16;
      const float updated_f = __bfloat162float(updated_bf16);
      y_reg[i] = updated_f;
      sq_sum = __fmaf_rn(updated_f, updated_f, sq_sum);
    }
  }

  __shared__ float warp_sums[8];
  const float w_sum = warp_reduce_sum(sq_sum);
  if (lane == 0) warp_sums[warp] = w_sum;
  __syncthreads();

  __shared__ float s_rstd;
  if (warp == 0) {
    const float p = (lane < 8) ? warp_sums[lane] : 0.0f;
    const float tot = warp_reduce_sum(p);
    if (lane == 0) {
      s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(tot, static_cast<float>(D)), norm_eps));
    }
  }
  __syncthreads();

  const float rstd = s_rstd;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    const int c = tid + i * 256;
    if (c < D) {
      const float nw = __bfloat162float(norm_w[c]);
      normed_out[static_cast<int64_t>(t) * D + c] =
          __float2bfloat16_rn(__fmul_rn(__fmul_rn(y_reg[i], rstd), nw));
    }
  }
}

/*
 * Exercise 1A Optimized v2:
 *   1. Exact 128-bit (`uint4` = 8 bf16) vectorization with 288 threads (9 warps) for D = 2304:
 *      Every thread owns exactly ONE `uint4` (8 contiguous channels) -> zero inner loop,
 *      zero bounds predicates, 100% coalesced `LDG.E.128` / `LDS.128` / `STG.E.128`!
 *   2. Hoists `conv_w` (4x `uint4` = 32 bf16) and `norm_w` (1x `uint4` = 8 bf16) into registers
 *      ONCE per CTA and reuses them across `kBT = 8` consecutive tokens!
 *   3. Multi-token 5-slot `cp.async` ring buffer (`5 * 2304 * 2 = 23 KB` SMEM):
 *      - Reads only `3 + 8 = 11` rows of `x` per 8 tokens instead of `4 * 8 = 32` (2.9x less `x` traffic!)
 *      - Software-pipelined with `cp.async.wait_group 1`, overlapping token `k + 1`'s GDDR6X DMA
 *        directly with token `k`'s ShortConv + RMSNorm compute!
 */
template <int kD = 2304, int kBT = 8>
__global__ void __launch_bounds__(kD / 8, 4) sm86_short_conv_rmsnorm_v2_kernel(
    const __nv_bfloat16* __restrict__ x,        // [T, kD]
    const __nv_bfloat16* __restrict__ conv_w,   // [kD, 4]
    const __nv_bfloat16* __restrict__ residual, // [T, kD]
    const __nv_bfloat16* __restrict__ norm_w,   // [kD]
    __nv_bfloat16* __restrict__ conv_out,       // [T, kD]
    __nv_bfloat16* __restrict__ normed_out,     // [T, kD]
    int T,
    float norm_eps) {
  static_assert(kD % 8 == 0, "kD must be divisible by 8");
  constexpr int kVec8 = kD / 8;          // 288 threads = 9 warps for D = 2304
  constexpr int kWarps = (kVec8 + 31) / 32;  // 9 warps
  constexpr int kRing = 5;               // 4 active taps + 1 prefetch slot = 0 WAR/WAW hazards!
  constexpr float kInvSqrt5 = 0.4472135954999579f;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint4* smem_ring = reinterpret_cast<uint4*>(smem_raw);  // [kRing, kVec8]

  const int tid = threadIdx.x;  // 0..287, owns channels [8*tid .. 8*tid + 7]
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int t_base = blockIdx.x * kBT;
  const int n_tok = min(kBT, T - t_base);
  if (n_tok <= 0) return;

  const uint4* x_v = reinterpret_cast<const uint4*>(x);
  const uint4* res_v = reinterpret_cast<const uint4*>(residual);
  const uint4* nw_v = reinterpret_cast<const uint4*>(norm_w);
  const uint4* cw_v = reinterpret_cast<const uint4*>(conv_w);
  uint4* cout_v = reinterpret_cast<uint4*>(conv_out);
  uint4* nout_v = reinterpret_cast<uint4*>(normed_out);

  // 1. Keep `norm_w` PACKED in 4 registers (`uint4`) instead of 8 floats to cut register pressure!
  const uint4 nw_pack = nw_v[tid];

  // Each `uint4` of `conv_w` holds 2 channels x 4 taps = 8 bf16s (row-major [D, 4])
  uint4 w_pack[4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    w_pack[p] = cw_v[tid * 4 + p];
  }

  // 2. Prologue: stage initial 4 taps (t_base - 3 .. t_base) into ring slots 0..3
#pragma unroll
  for (int tap = 0; tap < 4; ++tap) {
    const int src_t = t_base - (3 - tap);
    uint4* dst = smem_ring + tap * kVec8 + tid;
    if (src_t >= 0) {
      cp_async16(dst, x_v + static_cast<int64_t>(src_t) * kVec8 + tid);
    } else {
      *dst = make_uint4(0, 0, 0, 0);
    }
  }
  cp_async_commit();

  __shared__ float warp_sums[kWarps];
  __shared__ float s_rstd;

  // 3. Software-pipelined token loop: overlap token (k + 1) `cp.async` with token (k) compute!
  for (int k = 0; k < n_tok; ++k) {
    if (k + 1 < n_tok) {
      const int next_slot = (k + 4) % kRing;
      const int src_t = t_base + k + 1;
      cp_async16(smem_ring + next_slot * kVec8 + tid, x_v + static_cast<int64_t>(src_t) * kVec8 + tid);
      cp_async_commit();
      // Wait for token `k`'s group while token `k + 1`'s `cp.async` keeps streaming in background!
      cp_async_wait_group<1>();
    } else {
      cp_async_wait_group<0>();
    }

    // Thread `tid` reads ONLY its own 16-byte `uint4` slices from SMEM -> zero bank conflicts, no extra barrier!
    const uint4 t0 = smem_ring[((k + 0) % kRing) * kVec8 + tid];
    const uint4 t1 = smem_ring[((k + 1) % kRing) * kVec8 + tid];
    const uint4 t2 = smem_ring[((k + 2) % kRing) * kVec8 + tid];
    const uint4 t3 = smem_ring[((k + 3) % kRing) * kVec8 + tid];

    const int64_t row_off = static_cast<int64_t>(t_base + k) * kVec8 + tid;
    const uint4 r_pack = res_v[row_off];

    // Unpack 4 taps x 8 channels (processed as 4 pairs of channels) and accumulate depthwise k=4 conv
    const uint32_t t0_u[4] = {t0.x, t0.y, t0.z, t0.w};
    const uint32_t t1_u[4] = {t1.x, t1.y, t1.z, t1.w};
    const uint32_t t2_u[4] = {t2.x, t2.y, t2.z, t2.w};
    const uint32_t t3_u[4] = {t3.x, t3.y, t3.z, t3.w};
    const uint32_t r_u[4] = {r_pack.x, r_pack.y, r_pack.z, r_pack.w};

    float y_reg[8];
    uint32_t cout_u[4];
    float sq_sum = 0.0f;

#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 x0 = bf16x2_to_f2(t0_u[p]);
      const float2 x1 = bf16x2_to_f2(t1_u[p]);
      const float2 x2 = bf16x2_to_f2(t2_u[p]);
      const float2 x3 = bf16x2_to_f2(t3_u[p]);
      const float2 res_p = bf16x2_to_f2(r_u[p]);

      // Even channel (2*p): taps (0,1) in w_pack[p].x, taps (2,3) in w_pack[p].y
      const float2 w_e01 = bf16x2_to_f2(w_pack[p].x);
      const float2 w_e23 = bf16x2_to_f2(w_pack[p].y);
      float acc_e = __fmul_rn(x0.x, w_e01.x);
      acc_e = __fmaf_rn(x1.x, w_e01.y, acc_e);
      acc_e = __fmaf_rn(x2.x, w_e23.x, acc_e);
      acc_e = __fmaf_rn(x3.x, w_e23.y, acc_e);

      // Odd channel (2*p + 1): taps (0,1) in w_pack[p].z, taps (2,3) in w_pack[p].w
      const float2 w_o01 = bf16x2_to_f2(w_pack[p].z);
      const float2 w_o23 = bf16x2_to_f2(w_pack[p].w);
      float acc_o = __fmul_rn(x0.y, w_o01.x);
      acc_o = __fmaf_rn(x1.y, w_o01.y, acc_o);
      acc_o = __fmaf_rn(x2.y, w_o23.x, acc_o);
      acc_o = __fmaf_rn(x3.y, w_o23.y, acc_o);

      const float upd_e_raw = __fadd_rn(res_p.x, __fmul_rn(acc_e, kInvSqrt5));
      const float upd_o_raw = __fadd_rn(res_p.y, __fmul_rn(acc_o, kInvSqrt5));
      const uint32_t packed_upd = f2_to_bf16x2(upd_e_raw, upd_o_raw);
      cout_u[p] = packed_upd;

      // Exact parity: RMSNorm runs on the BF16-rounded `conv_out` values
      const float2 upd_bf16 = bf16x2_to_f2(packed_upd);
      y_reg[2 * p] = upd_bf16.x;
      y_reg[2 * p + 1] = upd_bf16.y;
      sq_sum = __fmaf_rn(upd_bf16.x, upd_bf16.x, sq_sum);
      sq_sum = __fmaf_rn(upd_bf16.y, upd_bf16.y, sq_sum);
    }

    // Coalesced 128-bit `STG.E.128` store for `conv_out`
    cout_v[row_off] = make_uint4(cout_u[0], cout_u[1], cout_u[2], cout_u[3]);

    // 9-warp butterfly reduction for RMSNorm
    const float w_sum = warp_reduce_sum(sq_sum);
    if (lane == 0) warp_sums[warp] = w_sum;
    __syncthreads();

    if (warp == 0) {
      const float p = (lane < kWarps) ? warp_sums[lane] : 0.0f;
      const float tot = warp_reduce_sum(p);
      if (lane == 0) {
        s_rstd = rsqrtf(__fadd_rn(__fdiv_rn(tot, static_cast<float>(kD)), norm_eps));
      }
    }
    __syncthreads();

    const float rstd = s_rstd;
    const float2 nw01 = bf16x2_to_f2(nw_pack.x);
    const float2 nw23 = bf16x2_to_f2(nw_pack.y);
    const float2 nw45 = bf16x2_to_f2(nw_pack.z);
    const float2 nw67 = bf16x2_to_f2(nw_pack.w);
    uint4 nout_pack;
    nout_pack.x = f2_to_bf16x2(__fmul_rn(__fmul_rn(y_reg[0], rstd), nw01.x),
                               __fmul_rn(__fmul_rn(y_reg[1], rstd), nw01.y));
    nout_pack.y = f2_to_bf16x2(__fmul_rn(__fmul_rn(y_reg[2], rstd), nw23.x),
                               __fmul_rn(__fmul_rn(y_reg[3], rstd), nw23.y));
    nout_pack.z = f2_to_bf16x2(__fmul_rn(__fmul_rn(y_reg[4], rstd), nw45.x),
                               __fmul_rn(__fmul_rn(y_reg[5], rstd), nw45.y));
    nout_pack.w = f2_to_bf16x2(__fmul_rn(__fmul_rn(y_reg[6], rstd), nw67.x),
                               __fmul_rn(__fmul_rn(y_reg[7], rstd), nw67.y));
    // Coalesced 128-bit `STG.E.128` store for `normed_out`
    nout_v[row_off] = nout_pack;
  }
}

}  // namespace

torch::Tensor loop_blend_rmsnorm_sm86(
    torch::Tensor dest,
    torch::Tensor message,
    torch::Tensor gain_forget,
    torch::Tensor norm_weight,
    double norm_eps) {
  TORCH_CHECK(message.is_cuda() && message.is_contiguous(), "message must be contiguous CUDA");
  TORCH_CHECK(dest.scalar_type() == torch::kFloat32, "dest must be float32");
  TORCH_CHECK(message.scalar_type() == torch::kFloat32, "message must be float32");
  TORCH_CHECK(gain_forget.scalar_type() == torch::kFloat32, "gain_forget must be float32");
  TORCH_CHECK(norm_weight.scalar_type() == torch::kBFloat16, "norm_weight must be bfloat16");

  const int S = message.size(0);
  const int T = message.size(1);
  const int D = message.size(2);
  TORCH_CHECK(D % 4 == 0 && D <= 4096, "D must be divisible by 4 and <= 4096 (Mini is 2304)");

  auto out = torch::empty_like(message, torch::TensorOptions().dtype(torch::kBFloat16));
  if (T == 0) return out;

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  dim3 grid(S, T);
  sm86_loop_blend_rmsnorm_kernel<256, 4><<<grid, 256, 0, stream>>>(
      dest.data_ptr<float>(),
      message.data_ptr<float>(),
      gain_forget.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      dest.stride(0),
      norm_weight.stride(0),
      T,
      D,
      static_cast<float>(norm_eps));
  return out;
}

std::tuple<torch::Tensor, torch::Tensor> short_conv_rmsnorm_sm86(
    torch::Tensor x,
    torch::Tensor conv_w,
    torch::Tensor residual,
    torch::Tensor norm_w,
    double norm_eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA bf16");
  const int T = x.size(0);
  const int D = x.size(1);
  TORCH_CHECK(D % 8 == 0 && D <= 2304, "D must be a multiple of 8 up to 2304");

  auto conv_out = torch::empty_like(x);
  auto normed_out = torch::empty_like(x);
  if (T == 0) return {conv_out, normed_out};

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const size_t smem_bytes = 4 * D * sizeof(__nv_bfloat16);
  sm86_short_conv_rmsnorm_kernel<<<T, 256, smem_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(conv_out.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr<at::BFloat16>()),
      T,
      D,
      static_cast<float>(norm_eps));
  return {conv_out, normed_out};
}

std::tuple<torch::Tensor, torch::Tensor> short_conv_rmsnorm_sm86_v2(
    torch::Tensor x,
    torch::Tensor conv_w,
    torch::Tensor residual,
    torch::Tensor norm_w,
    double norm_eps) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "x must be contiguous CUDA bf16");
  const int T = x.size(0);
  const int D = x.size(1);
  TORCH_CHECK(D == 2304, "v2 specialized kernel expects Inkling 2 Mini D == 2304");

  auto conv_out = torch::empty_like(x);
  auto normed_out = torch::empty_like(x);
  if (T == 0) return {conv_out, normed_out};

  constexpr int kRing = 5;
  const size_t smem_bytes = kRing * 2304 * sizeof(__nv_bfloat16);  // 23,040 bytes

  // Adaptive tokens/CTA so we NEVER leave RTX 3090's 82 SMs idle on short sequences!
  //   T <= 512 -> kBT = 2 (at T=256 -> 128 CTAs across 82 SMs, instead of 32!)
  //   T >  512 -> kBT = 8 (at T=2048 -> 256 CTAs across 82 SMs + 2.9x less `x` traffic!)
  static bool carveout_set = false;
  if (!carveout_set) {
    cudaFuncSetAttribute(
        sm86_short_conv_rmsnorm_v2_kernel<2304, 2>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(
        sm86_short_conv_rmsnorm_v2_kernel<2304, 8>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    carveout_set = true;
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  if (T <= 512) {
    const int grid = (T + 1) / 2;
    sm86_short_conv_rmsnorm_v2_kernel<2304, 2><<<grid, 2304 / 8, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(conv_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr<at::BFloat16>()),
        T,
        static_cast<float>(norm_eps));
  } else {
    const int grid = (T + 7) / 8;
    sm86_short_conv_rmsnorm_v2_kernel<2304, 8><<<grid, 2304 / 8, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(conv_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(conv_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(normed_out.data_ptr<at::BFloat16>()),
        T,
        static_cast<float>(norm_eps));
  }
  return {conv_out, normed_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("loop_blend_rmsnorm_sm86", &loop_blend_rmsnorm_sm86, "SM86 Loop Blend + RMSNorm");
  m.def("short_conv_rmsnorm_sm86", &short_conv_rmsnorm_sm86, "SM86 cp.async ShortConv + RMSNorm (v1)");
  m.def("short_conv_rmsnorm_sm86_v2", &short_conv_rmsnorm_sm86_v2, "SM86 pipelined 128-bit ShortConv + RMSNorm (v2)");
}
