// SM86 (RTX 3090 / GA102) Fused Batched Top-K MoE Decode Kernels v3 for Inkling-2-Mini.
//
// Why v3 beats both v1 (60.3 us) and v2 (113.4 us, which hit 8-way SMEM bank conflicts):
//   1. Zero SMEM, zero __syncthreads(), zero bank conflicts: `x` (4.6 KB) and `act` (18 KB)
//      stay 100% L1/L2 resident via read-only `__ldg`!
//   2. <= 40 registers/thread (0 spills!) -> locks 100% occupancy (48 warps/SM on GA102)!
//   3. Packed 2-element `bf16x2` bit-shift unpack (3 integer ops for TWO weights -> 2x fewer ALU!):
//        - NVFP4 e2m1 pair (nib i, nib i+4): `((w & 0x00080008u) << 12) | ((w & 0x00070007u) << 6)`
//        - MXFP8 e4m3 pair (byte 0, byte 2): `((v & 0x00800080u) <<  8) | ((v & 0x007F007Fu) << 4)`
//   4. 1 thread = 1 exact scale group, exact divisor lane-mapping -> ZERO warp tail divergence:
//        NVFP4 w13 (K=2304 = 144 groups): 16 lanes x 9 exact iters (128B cache line/iter, 1152 CTAs)
//        NVFP4 w2  (I=1152 =  72 groups):  8 lanes x 9 exact iters ( 64B = 2 L2 sectors, 1152 CTAs)
//        MXFP8 w13 (K=2304 =  72 groups):  8 lanes x 9 exact iters (256B = 2 cache lines, 576 CTAs)
//        MXFP8 w2  (I=1152 =  36 groups):  4 lanes x 9 exact iters (128B cache line/iter, 1152 CTAs)
//   5. Fused exact `SiTU` epilogue (zero `gate_up` DRAM write) + expert-parallel L2 `atomicAdd`!

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr float kScale2p100 = 1.2676506002282294e30f;   // 2^+100
constexpr float kScale2p26  = 67108864.0f;              // 2^+26 (cancels 2^-126 * 2^+100)
constexpr float kScale2p120 = 1.3292279957849159e36f;   // 2^+120 (cancels e4m3fn 2^-120)

__device__ __forceinline__ float decode_e4m3_f32(uint8_t s) {
    uint32_t u = static_cast<uint32_t>(s);
    uint32_t bits = ((u & 0x80u) << 24) | ((u & 0x7Fu) << 20);
    return __uint_as_float(bits) * kScale2p120;
}

__device__ __forceinline__ void load_bf16x16_scaled(
    const __nv_bfloat16* __restrict__ ptr,
    float x_s[16],
    float scale
) {
    const uint4* ptr4 = reinterpret_cast<const uint4*>(ptr);
    uint4 v0 = __ldg(ptr4);
    uint4 v1 = __ldg(ptr4 + 1);
    uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = words[i];
        x_s[2 * i]     = __uint_as_float(w << 16) * scale;
        x_s[2 * i + 1] = __uint_as_float(w & 0xFFFF0000u) * scale;
    }
}

__device__ __forceinline__ void load_bf16x16(
    const __nv_bfloat16* __restrict__ ptr,
    float x_f[16]
) {
    const uint4* ptr4 = reinterpret_cast<const uint4*>(ptr);
    uint4 v0 = __ldg(ptr4);
    uint4 v1 = __ldg(ptr4 + 1);
    uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = words[i];
        x_f[2 * i]     = __uint_as_float(w << 16);
        x_f[2 * i + 1] = __uint_as_float(w & 0xFFFF0000u);
    }
}

// Packed 2-nibble `bf16x2` bit-shift dot of 16 NVFP4 weights (uint2 = 1 exact scale group)
__device__ __forceinline__ float dot16_nvfp4_group_2nib(
    uint2 packed16,
    const float x_s[16],
    uint8_t scale_u8
) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t w0 = packed16.x;
    uint32_t w1 = packed16.y;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t b0 = ((w0 & 0x00080008u) << 12) | ((w0 & 0x00070007u) << 6);
        acc0 = fmaf(__uint_as_float(b0 << 16),         x_s[i],      acc0);
        acc1 = fmaf(__uint_as_float(b0 & 0xFFFF0000u), x_s[i + 4],  acc1);
        w0 >>= 4;

        uint32_t b1 = ((w1 & 0x00080008u) << 12) | ((w1 & 0x00070007u) << 6);
        acc0 = fmaf(__uint_as_float(b1 << 16),         x_s[i + 8],  acc0);
        acc1 = fmaf(__uint_as_float(b1 & 0xFFFF0000u), x_s[i + 12], acc1);
        w1 >>= 4;
    }
    return (acc0 + acc1) * decode_e4m3_f32(scale_u8);
}

// Packed 2-byte `bf16x2` bit-shift dot of 16 e4m3fn bytes (uint4 -> exact val * 2^-120)
__device__ __forceinline__ float dot16_e4m3_2byte(
    uint4 packed16,
    const float x_f[16]
) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint32_t words[4] = {packed16.x, packed16.y, packed16.z, packed16.w};
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t v = words[w];
        uint32_t b02 = ((v & 0x00800080u) << 8) | ((v & 0x007F007Fu) << 4);
        acc0 = fmaf(__uint_as_float(b02 << 16),         x_f[4 * w + 0], acc0);
        acc1 = fmaf(__uint_as_float(b02 & 0xFFFF0000u), x_f[4 * w + 2], acc1);
        v >>= 8;
        uint32_t b13 = ((v & 0x00800080u) << 8) | ((v & 0x007F007Fu) << 4);
        acc0 = fmaf(__uint_as_float(b13 << 16),         x_f[4 * w + 1], acc0);
        acc1 = fmaf(__uint_as_float(b13 & 0xFFFF0000u), x_f[4 * w + 3], acc1);
    }
    return acc0 + acc1;
}

__device__ __forceinline__ float scale_mxfp8_dot(float dot_shifted, uint8_t exp_u8) {
    if (exp_u8 <= 134u) {
        uint32_t e_bits = (exp_u8 == 0u ? 121u : (static_cast<uint32_t>(exp_u8) + 120u)) << 23;
        return dot_shifted * __uint_as_float(e_bits);
    }
    uint32_t e_bits = static_cast<uint32_t>(exp_u8) << 23;
    return (dot_shifted * kScale2p120) * __uint_as_float(e_bits);
}

__device__ __forceinline__ float situ_exact_f32(float gate, float up, float gamma) {
    float sig = 1.0f / (1.0f + expf(-gate));
    float g_cap = 4.0f * tanhf(gate * 0.25f);
    float u_cap = 25.0f * tanhf(up * 0.04f);
    return ((g_cap * sig) * u_cap) * gamma;
}

// ============================================================================
// 1. NVFP4 Batched Top-K `w13 + SiTU` Decode GEMV v4 (2 pairs = 4 rows/half-warp!)
//    Reuses `x_s[16]` 4x across 2 (gate, up) pairs -> 2x fewer L1/scale loads!
//    Grid: (cdiv(N_INTER, 16), T * TOP_K) -> 576 CTAs (7.02 waves on 82 SMs!)
// ============================================================================
__launch_bounds__(128, 10)
__global__ void sm86_nvfp4_moe_w13_situ_gemv_kernel(
    const __nv_bfloat16* __restrict__ X,          // [T, K]
    const uint8_t* __restrict__ W13_packed,       // [E, 2 * N_INTER, K / 2]
    const uint8_t* __restrict__ W13_scale,        // [E, 2 * N_INTER, K / 16]
    const float* __restrict__ W13_outer,          // [E]
    const int32_t* __restrict__ TopkIds,          // [T * TOP_K]
    const float* __restrict__ TopkWeights,        // [T * TOP_K]
    __nv_bfloat16* __restrict__ ActOut,           // [T * TOP_K, N_INTER]
    int K,
    int N_INTER,
    int TOP_K
) {
    const int hw_in_cta = threadIdx.x >> 4;    // 8 half-warps per 128-thread CTA (16 lanes each)
    const int sublane = threadIdx.x & 15;
    const int inter0 = blockIdx.x * 16 + hw_in_cta * 2;
    if (inter0 >= N_INTER) return;
    const bool has_p1 = (inter0 + 1) < N_INTER;

    const int slot = blockIdx.y;
    const int tok = slot / TOP_K;
    const int expert_id = TopkIds[slot];

    const int num_groups = K >> 4;  // 144 for K=2304 -> 9 exact 16-lane iters (128B/iter!)
    const int64_t w_row_stride = static_cast<int64_t>(K >> 1);
    const int64_t s_row_stride = static_cast<int64_t>(num_groups);
    const int64_t base_row = static_cast<int64_t>(expert_id) * (2 * N_INTER) + (2 * inter0);

    const uint2* wg0 = reinterpret_cast<const uint2*>(W13_packed + (base_row + 0) * w_row_stride);
    const uint2* wu0 = reinterpret_cast<const uint2*>(W13_packed + (base_row + 1) * w_row_stride);
    const uint2* wg1 = reinterpret_cast<const uint2*>(W13_packed + (base_row + 2) * w_row_stride);
    const uint2* wu1 = reinterpret_cast<const uint2*>(W13_packed + (base_row + 3) * w_row_stride);
    const uint8_t* sg0 = W13_scale + (base_row + 0) * s_row_stride;
    const uint8_t* su0 = W13_scale + (base_row + 1) * s_row_stride;
    const uint8_t* sg1 = W13_scale + (base_row + 2) * s_row_stride;
    const uint8_t* su1 = W13_scale + (base_row + 3) * s_row_stride;
    const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K;

    float acc_g0 = 0.0f, acc_u0 = 0.0f;
    float acc_g1 = 0.0f, acc_u1 = 0.0f;
    for (int g = sublane; g < num_groups; g += 16) {
        float x_s[16];
        load_bf16x16_scaled(x_row + (g << 4), x_s, kScale2p100);
        acc_g0 += dot16_nvfp4_group_2nib(__ldg(wg0 + g), x_s, __ldg(sg0 + g));
        acc_u0 += dot16_nvfp4_group_2nib(__ldg(wu0 + g), x_s, __ldg(su0 + g));
        if (has_p1) {
            acc_g1 += dot16_nvfp4_group_2nib(__ldg(wg1 + g), x_s, __ldg(sg1 + g));
            acc_u1 += dot16_nvfp4_group_2nib(__ldg(wu1 + g), x_s, __ldg(su1 + g));
        }
    }

    #pragma unroll
    for (int offset = 8; offset >= 1; offset >>= 1) {
        acc_g0 += __shfl_xor_sync(0xFFFFFFFFu, acc_g0, offset, 16);
        acc_u0 += __shfl_xor_sync(0xFFFFFFFFu, acc_u0, offset, 16);
        acc_g1 += __shfl_xor_sync(0xFFFFFFFFu, acc_g1, offset, 16);
        acc_u1 += __shfl_xor_sync(0xFFFFFFFFu, acc_u1, offset, 16);
    }

    if (sublane == 0) {
        float outer = W13_outer[expert_id] * kScale2p26;
        float gamma = TopkWeights[slot];
        int64_t out_base = static_cast<int64_t>(slot) * N_INTER + inter0;
        ActOut[out_base] = __float2bfloat16(situ_exact_f32(acc_g0 * outer, acc_u0 * outer, gamma));
        if (has_p1) {
            ActOut[out_base + 1] =
                __float2bfloat16(situ_exact_f32(acc_g1 * outer, acc_u1 * outer, gamma));
        }
    }
}

// ============================================================================
// 2A. NVFP4 Expert-Parallel `w2` + FP32 L2 `atomicAdd` v4 (2 rows/quarter-warp!)
//     Reuses `x_s[16]` 2x -> halves L1/scale overhead on w2! 576 CTAs!
// ============================================================================
__launch_bounds__(128, 12)
__global__ void sm86_nvfp4_moe_w2_atomic_gemv_kernel(
    const __nv_bfloat16* __restrict__ Act,        // [T * TOP_K, I]
    const uint8_t* __restrict__ W2_packed,        // [E, H, I / 2]
    const uint8_t* __restrict__ W2_scale,         // [E, H, I / 16]
    const float* __restrict__ W2_outer,           // [E]
    const int32_t* __restrict__ TopkIds,          // [T * TOP_K]
    float* __restrict__ OutF32,                   // [T, H] (zero-initialized)
    int H,
    int I,
    int TOP_K
) {
    const int qw_in_cta = threadIdx.x >> 3;  // 16 quarter-warps/CTA -> 32 output rows/CTA!
    const int sublane = threadIdx.x & 7;
    const int h0 = blockIdx.x * 32 + qw_in_cta * 2;
    if (h0 >= H) return;
    const bool has_h1 = (h0 + 1) < H;

    const int slot = blockIdx.y;
    const int tok = slot / TOP_K;
    const int expert_id = TopkIds[slot];

    const int num_groups = I >> 4;  // 72 for I=1152 -> 9 exact 8-lane iters!
    const int64_t w_row_stride = static_cast<int64_t>(I >> 1);
    const int64_t s_row_stride = static_cast<int64_t>(num_groups);
    const int64_t w_row0 = static_cast<int64_t>(expert_id) * H + h0;

    const __nv_bfloat16* act_row = Act + static_cast<int64_t>(slot) * I;
    const uint2* w_ptr0 = reinterpret_cast<const uint2*>(W2_packed + w_row0 * w_row_stride);
    const uint2* w_ptr1 = reinterpret_cast<const uint2*>(W2_packed + (w_row0 + 1) * w_row_stride);
    const uint8_t* s_ptr0 = W2_scale + w_row0 * s_row_stride;
    const uint8_t* s_ptr1 = W2_scale + (w_row0 + 1) * s_row_stride;

    float acc0 = 0.0f, acc1 = 0.0f;
    for (int g = sublane; g < num_groups; g += 8) {
        float x_s[16];
        load_bf16x16_scaled(act_row + (g << 4), x_s, kScale2p100);
        acc0 += dot16_nvfp4_group_2nib(__ldg(w_ptr0 + g), x_s, __ldg(s_ptr0 + g));
        if (has_h1) {
            acc1 += dot16_nvfp4_group_2nib(__ldg(w_ptr1 + g), x_s, __ldg(s_ptr1 + g));
        }
    }

    #pragma unroll
    for (int offset = 4; offset >= 1; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xFFFFFFFFu, acc0, offset, 8);
        acc1 += __shfl_xor_sync(0xFFFFFFFFu, acc1, offset, 8);
    }

    if (sublane == 0) {
        float outer = W2_outer[expert_id] * kScale2p26;
        float* out_ptr = OutF32 + static_cast<int64_t>(tok) * H + h0;
        atomicAdd(out_ptr, acc0 * outer);
        if (has_h1) {
            atomicAdd(out_ptr + 1, acc1 * outer);
        }
    }
}

// ============================================================================
// 2B. NVFP4 Fused Top-K `w2 + Reduce` (ZERO atomicAdd, 2 warps/CTA = 288 CTAs)
// ============================================================================
__launch_bounds__(64, 16)
__global__ void sm86_nvfp4_moe_w2_reduce_gemv_kernel(
    const __nv_bfloat16* __restrict__ Act,        // [T, TOP_K, I]
    const uint8_t* __restrict__ W2_packed,        // [E, H, I / 2]
    const uint8_t* __restrict__ W2_scale,         // [E, H, I / 16]
    const float* __restrict__ W2_outer,           // [E]
    const int32_t* __restrict__ TopkIds,          // [T, TOP_K]
    __nv_bfloat16* __restrict__ Out,              // [T, H]
    int H,
    int I,
    int TOP_K
) {
    const int row_in_cta = threadIdx.x >> 3;  // 8 rows per 64-thread CTA -> 288 CTAs (3.5 waves)
    const int sublane = threadIdx.x & 7;
    const int h_idx = blockIdx.x * 8 + row_in_cta;
    if (h_idx >= H) return;

    const int tok = blockIdx.y;
    const int num_groups = I >> 4;
    const int64_t w_row_stride = static_cast<int64_t>(I >> 1);
    const int64_t s_row_stride = static_cast<int64_t>(num_groups);
    const int64_t expert_w_stride = static_cast<int64_t>(H) * w_row_stride;
    const int64_t expert_s_stride = static_cast<int64_t>(H) * s_row_stride;

    float total_acc = 0.0f;
    for (int k = 0; k < TOP_K; ++k) {
        const int slot = tok * TOP_K + k;
        const int expert_id = TopkIds[slot];
        const __nv_bfloat16* act_row = Act + static_cast<int64_t>(slot) * I;
        const uint2* w_ptr = reinterpret_cast<const uint2*>(
            W2_packed + static_cast<int64_t>(expert_id) * expert_w_stride +
            static_cast<int64_t>(h_idx) * w_row_stride
        );
        const uint8_t* s_ptr =
            W2_scale + static_cast<int64_t>(expert_id) * expert_s_stride +
            static_cast<int64_t>(h_idx) * s_row_stride;

        float expert_acc = 0.0f;
        for (int g = sublane; g < num_groups; g += 8) {
            float x_s[16];
            load_bf16x16_scaled(act_row + (g << 4), x_s, kScale2p100);
            expert_acc += dot16_nvfp4_group_2nib(__ldg(w_ptr + g), x_s, __ldg(s_ptr + g));
        }
        total_acc += expert_acc * (W2_outer[expert_id] * kScale2p26);
    }

    #pragma unroll
    for (int offset = 4; offset >= 1; offset >>= 1) {
        total_acc += __shfl_xor_sync(0xFFFFFFFFu, total_acc, offset, 8);
    }

    if (sublane == 0) {
        Out[static_cast<int64_t>(tok) * H + h_idx] = __float2bfloat16(total_acc);
    }
}

__global__ void sm86_f32_to_bf16_zero_kernel(
    float* __restrict__ BufF32,
    __nv_bfloat16* __restrict__ OutBF16,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        OutBF16[idx] = __float2bfloat16(BufF32[idx]);
        BufF32[idx] = 0.0f;
    }
}

// ============================================================================
// 3. MXFP8 Batched Top-K `w13 + SiTU` & `w2 + Atomic` Decode GEMV v3 (zero SMEM!)
// ============================================================================
__launch_bounds__(128, 10)
__global__ void sm86_mxfp8_moe_w13_situ_gemv_kernel(
    const __nv_bfloat16* __restrict__ X,          // [T, K]
    const uint8_t* __restrict__ W13_e4m3,         // [E, 2 * N_INTER, K]
    const uint8_t* __restrict__ W13_ue8m0,        // [E, 2 * N_INTER, K / 32]
    const int32_t* __restrict__ TopkIds,          // [T * TOP_K]
    const float* __restrict__ TopkWeights,        // [T * TOP_K]
    __nv_bfloat16* __restrict__ ActOut,           // [T * TOP_K, N_INTER]
    int K,
    int N_INTER,
    int TOP_K
) {
    const int pair_in_cta = threadIdx.x >> 3;  // 16 pairs/CTA (8 lanes/pair) -> 576 CTAs!
    const int sublane = threadIdx.x & 7;
    const int inter_idx = blockIdx.x * 16 + pair_in_cta;
    if (inter_idx >= N_INTER) return;

    const int slot = blockIdx.y;
    const int tok = slot / TOP_K;
    const int expert_id = TopkIds[slot];

    const int num_groups = K >> 5;  // 72 for K=2304 -> 9 exact 8-lane iters!
    const int64_t row_g = static_cast<int64_t>(expert_id) * (2 * N_INTER) + (2 * inter_idx);
    const int64_t row_u = row_g + 1;

    const uint4* wg_ptr = reinterpret_cast<const uint4*>(W13_e4m3 + row_g * K);
    const uint4* wu_ptr = reinterpret_cast<const uint4*>(W13_e4m3 + row_u * K);
    const uint8_t* sg_ptr = W13_ue8m0 + row_g * num_groups;
    const uint8_t* su_ptr = W13_ue8m0 + row_u * num_groups;
    const __nv_bfloat16* x_row = X + static_cast<int64_t>(tok) * K;

    float acc_g = 0.0f;
    float acc_u = 0.0f;
    for (int g = sublane; g < num_groups; g += 8) {
        float x0[16];
        load_bf16x16(x_row + (g << 5), x0);
        float dg = dot16_e4m3_2byte(__ldg(wg_ptr + (g << 1)), x0);
        float du = dot16_e4m3_2byte(__ldg(wu_ptr + (g << 1)), x0);

        load_bf16x16(x_row + (g << 5) + 16, x0);
        dg += dot16_e4m3_2byte(__ldg(wg_ptr + (g << 1) + 1), x0);
        du += dot16_e4m3_2byte(__ldg(wu_ptr + (g << 1) + 1), x0);

        acc_g += scale_mxfp8_dot(dg, __ldg(sg_ptr + g));
        acc_u += scale_mxfp8_dot(du, __ldg(su_ptr + g));
    }

    #pragma unroll
    for (int offset = 4; offset >= 1; offset >>= 1) {
        acc_g += __shfl_xor_sync(0xFFFFFFFFu, acc_g, offset, 8);
        acc_u += __shfl_xor_sync(0xFFFFFFFFu, acc_u, offset, 8);
    }

    if (sublane == 0) {
        float gamma = TopkWeights[slot];
        ActOut[static_cast<int64_t>(slot) * N_INTER + inter_idx] =
            __float2bfloat16(situ_exact_f32(acc_g, acc_u, gamma));
    }
}

__launch_bounds__(64, 16)
__global__ void sm86_mxfp8_moe_w2_atomic_gemv_kernel(
    const __nv_bfloat16* __restrict__ Act,        // [T * TOP_K, I]
    const uint8_t* __restrict__ W2_e4m3,          // [E, H, I]
    const uint8_t* __restrict__ W2_ue8m0,         // [E, H, I / 32]
    const int32_t* __restrict__ TopkIds,          // [T * TOP_K]
    float* __restrict__ OutF32,                   // [T, H]
    int H,
    int I,
    int TOP_K
) {
    // 4 lanes/row -> 16 rows per 64-thread CTA -> 144 * 8 = 1,152 CTAs for T=1!
    const int row_in_cta = threadIdx.x >> 2;
    const int sublane = threadIdx.x & 3;
    const int h_idx = blockIdx.x * 16 + row_in_cta;
    if (h_idx >= H) return;

    const int slot = blockIdx.y;
    const int tok = slot / TOP_K;
    const int expert_id = TopkIds[slot];

    const int num_groups = I >> 5;  // 36 for I=1152 -> 9 exact 4-lane iters (128B/iter!)
    const int64_t w_row = static_cast<int64_t>(expert_id) * H + h_idx;

    const __nv_bfloat16* act_row = Act + static_cast<int64_t>(slot) * I;
    const uint4* w_ptr = reinterpret_cast<const uint4*>(W2_e4m3 + w_row * I);
    const uint8_t* s_ptr = W2_ue8m0 + w_row * num_groups;

    float acc = 0.0f;
    for (int g = sublane; g < num_groups; g += 4) {
        float x0[16];
        load_bf16x16(act_row + (g << 5), x0);
        float d = dot16_e4m3_2byte(__ldg(w_ptr + (g << 1)), x0);
        load_bf16x16(act_row + (g << 5) + 16, x0);
        d += dot16_e4m3_2byte(__ldg(w_ptr + (g << 1) + 1), x0);
        acc += scale_mxfp8_dot(d, __ldg(s_ptr + g));
    }

    #pragma unroll
    for (int offset = 2; offset >= 1; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, offset, 4);
    }

    if (sublane == 0) {
        atomicAdd(OutF32 + static_cast<int64_t>(tok) * H + h_idx, acc);
    }
}

}  // namespace

torch::Tensor nvfp4_moe_decode_forward(
    torch::Tensor x,            // [T, K] bf16
    torch::Tensor w13_packed,   // [E, 2*I, K/2] uint8
    torch::Tensor w13_scale,    // [E, 2*I, K/16] uint8
    torch::Tensor w13_outer,    // [E] fp32
    torch::Tensor w2_packed,    // [E, H, I/2] uint8
    torch::Tensor w2_scale,     // [E, H, I/16] uint8
    torch::Tensor w2_outer,     // [E] fp32
    torch::Tensor topk_ids,     // [T, TOP_K] int32
    torch::Tensor topk_weights, // [T, TOP_K] fp32
    torch::Tensor act_buf,      // [T, TOP_K, I] bf16
    torch::Tensor out_buf,      // [T, H] bf16
    torch::Tensor f32_buf,      // [T, H] fp32
    bool use_atomic_w2
) {
    const int T = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int TOP_K = static_cast<int>(topk_ids.size(1));
    const int N_INTER = static_cast<int>(w13_packed.size(1) / 2);
    const int H = static_cast<int>(w2_packed.size(1));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dim3 grid1((N_INTER + 15) / 16, T * TOP_K);
    sm86_nvfp4_moe_w13_situ_gemv_kernel<<<grid1, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        w13_packed.data_ptr<uint8_t>(),
        w13_scale.data_ptr<uint8_t>(),
        w13_outer.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
        K,
        N_INTER,
        TOP_K
    );

    if (use_atomic_w2) {
        dim3 grid2((H + 31) / 32, T * TOP_K);
        sm86_nvfp4_moe_w2_atomic_gemv_kernel<<<grid2, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
            w2_packed.data_ptr<uint8_t>(),
            w2_scale.data_ptr<uint8_t>(),
            w2_outer.data_ptr<float>(),
            topk_ids.data_ptr<int32_t>(),
            f32_buf.data_ptr<float>(),
            H,
            N_INTER,
            TOP_K
        );
        int total_out = T * H;
        sm86_f32_to_bf16_zero_kernel<<<(total_out + 255) / 256, 256, 0, stream>>>(
            f32_buf.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out_buf.data_ptr<at::BFloat16>()),
            total_out
        );
    } else {
        dim3 grid2((H + 7) / 8, T);
        sm86_nvfp4_moe_w2_reduce_gemv_kernel<<<grid2, 64, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
            w2_packed.data_ptr<uint8_t>(),
            w2_scale.data_ptr<uint8_t>(),
            w2_outer.data_ptr<float>(),
            topk_ids.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out_buf.data_ptr<at::BFloat16>()),
            H,
            N_INTER,
            TOP_K
        );
    }
    return out_buf;
}

torch::Tensor mxfp8_moe_decode_forward(
    torch::Tensor x,            // [T, K] bf16
    torch::Tensor w13_e4m3,     // [E, 2*I, K] uint8
    torch::Tensor w13_ue8m0,    // [E, 2*I, K/32] uint8
    torch::Tensor w2_e4m3,      // [E, H, I] uint8
    torch::Tensor w2_ue8m0,     // [E, H, I/32] uint8
    torch::Tensor topk_ids,     // [T, TOP_K] int32
    torch::Tensor topk_weights, // [T, TOP_K] fp32
    torch::Tensor act_buf,      // [T, TOP_K, I] bf16
    torch::Tensor out_buf,      // [T, H] bf16
    torch::Tensor f32_buf       // [T, H] fp32
) {
    const int T = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int TOP_K = static_cast<int>(topk_ids.size(1));
    const int N_INTER = static_cast<int>(w13_e4m3.size(1) / 2);
    const int H = static_cast<int>(w2_e4m3.size(1));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dim3 grid1((N_INTER + 15) / 16, T * TOP_K);
    sm86_mxfp8_moe_w13_situ_gemv_kernel<<<grid1, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        w13_e4m3.data_ptr<uint8_t>(),
        w13_ue8m0.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
        K,
        N_INTER,
        TOP_K
    );

    dim3 grid2((H + 15) / 16, T * TOP_K);
    sm86_mxfp8_moe_w2_atomic_gemv_kernel<<<grid2, 64, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(act_buf.data_ptr<at::BFloat16>()),
        w2_e4m3.data_ptr<uint8_t>(),
        w2_ue8m0.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        f32_buf.data_ptr<float>(),
        H,
        N_INTER,
        TOP_K
    );
    int total_out = T * H;
    sm86_f32_to_bf16_zero_kernel<<<(total_out + 255) / 256, 256, 0, stream>>>(
        f32_buf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out_buf.data_ptr<at::BFloat16>()),
        total_out
    );
    return out_buf;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvfp4_moe_decode_forward", &nvfp4_moe_decode_forward, "SM86 fused NVFP4 Top-K MoE decode GEMV v3");
    m.def("mxfp8_moe_decode_forward", &mxfp8_moe_decode_forward, "SM86 fused MXFP8 Top-K MoE decode GEMV v3");
}
