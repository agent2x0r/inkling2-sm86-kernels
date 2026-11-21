#!/usr/bin/env python3
"""Lab 3: Sub-Byte Tensor Core GEMM — W4A16 `NVFP4` & W8A16 `MXFP8` + `SiTU` on Ampere (SM86).

Run:
    uv run python labs/lab3_sm86_nvfp4_mxfp8_moe.py

The Hardware Challenge on RTX 3090 (`SM86`):
- Official `thinkingmachines/Inkling-2-Mini` stores 89% of its parameters in routed experts:
    * Layers  0..19: `NVFP4` (`uint8` packing two `e2m1` nibbles + `float8_e4m3fn` per-16 scales
      + `amax / 2688.0` outer FP32 scale)
    * Layers 20..23: `MXFP8` (`float8_e4m3fn` weights + `uint8` `UE8M0` per-32 power-of-two scales)
- Ampere has NO hardware FP4 or FP8 Tensor Cores (`tl.dot_scaled` requires SM100/SM120!).
- Instead, we keep weights packed in VRAM (saving 3.5x memory bandwidth on GDDR6X!) and unpack
  `e2m1` / `e4m3` into `bfloat16` IN REGISTERS inside the inner K-tile loop right before Ampere's
  `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`tl.dot`)!

The 2-Cycle PTX `prmt.b32` Byte-Permute Trick for `e2m1 -> bf16`:
Look at all 8 positive `e2m1` magnitudes (`mag = nibble & 0x7`) as IEEE-754 `bfloat16` hex words:
    mag:    0       1       2       3       4       5       6       7
    val:   0.0     0.5     1.0     1.5     2.0     3.0     4.0     6.0
    bf16: 0x0000  0x3F00  0x3F80  0x3FC0  0x4000  0x4040  0x4080  0x40C0
Notice:
  - The 8 low bytes  `[0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0]` fit in TWO `uint32` registers!
  - The 8 high bytes `[0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40]` fit in TWO `uint32` registers!
  - Ampere's single-cycle `prmt.b32 d, lut_lo, lut_hi, selector` selects ANY 4 bytes from an 8-byte
    table in ONE instruction—decoding `e2m1` nibbles to exact `bf16` with ZERO branches and ZERO
    shared-memory bank conflicts!
"""

from __future__ import annotations

import torch

from inkling_sm86.checkpoint_slice import (
    fetch_real_mxfp8_expert_sample,
    fetch_real_nvfp4_expert_sample,
)
from inkling_sm86.reference import (
    E2M1_LUT,
    dequant_mxfp8_weight,
    dequant_nvfp4_weight,
    expert_mlp_ref,
    situ_activation_ref,
)


def bitwise_e2m1_to_bf16_cpu_reference(packed_u8: torch.Tensor) -> torch.Tensor:
    """Pure integer bit-exact proof of the `prmt.b32` nibble-LUT algorithm on CPU."""
    lo_lut = torch.tensor([0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0], dtype=torch.int32)
    hi_lut = torch.tensor([0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40], dtype=torch.int32)

    def decode_nibble(nib: torch.Tensor) -> torch.Tensor:
        mag = (nib & 0x07).long()
        sign_bit = (nib.to(torch.int32) & 0x08) << 12
        bf16_i16 = (sign_bit | (hi_lut[mag] << 8) | lo_lut[mag]).to(torch.int16)
        return bf16_i16.view(torch.bfloat16)

    even = decode_nibble(packed_u8 & 0x0F)
    odd = decode_nibble(packed_u8 >> 4)
    return torch.stack((even, odd), dim=-1).flatten(-2)


def build_sm86_moe_triton_kernels():
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _unpack_e2m1_pair_ptx(packed_u8):
        """Unpack `uint8` (`low nibble = even, high nibble = odd`) -> two `float32` values via PTX `prmt.b32`."""
        return tl.inline_asm_elementwise(
            """
            {
                .reg .b32 nib_lo, nib_hi, mag_lo, mag_hi, sgn_lo, sgn_hi;
                .reg .b32 b_lo, b_hi, bf_lo, bf_hi;
                and.b32 nib_lo, $2, 0x0f;
                shr.u32 nib_hi, $2, 4;
                and.b32 nib_hi, nib_hi, 0x0f;
                and.b32 mag_lo, nib_lo, 0x07;
                and.b32 mag_hi, nib_hi, 0x07;
                and.b32 sgn_lo, nib_lo, 0x08;
                and.b32 sgn_hi, nib_hi, 0x08;
                shl.b32 sgn_lo, sgn_lo, 28;
                shl.b32 sgn_hi, sgn_hi, 28;
                prmt.b32 b_lo, 0xc0800000, 0xc0804000, mag_lo;
                prmt.b32 b_hi, 0x3f3f3f00, 0x40404040, mag_lo;
                and.b32 b_lo, b_lo, 0xff;
                and.b32 b_hi, b_hi, 0xff;
                shl.b32 bf_lo, b_hi, 24;
                shl.b32 b_lo, b_lo, 16;
                or.b32 bf_lo, bf_lo, b_lo;
                or.b32 $0, bf_lo, sgn_lo;

                prmt.b32 b_lo, 0xc0800000, 0xc0804000, mag_hi;
                prmt.b32 b_hi, 0x3f3f3f00, 0x40404040, mag_hi;
                and.b32 b_lo, b_lo, 0xff;
                and.b32 b_hi, b_hi, 0xff;
                shl.b32 bf_hi, b_hi, 24;
                shl.b32 b_lo, b_lo, 16;
                or.b32 bf_hi, bf_hi, b_lo;
                or.b32 $1, bf_hi, sgn_hi;
            }
            """,
            "=f,=f,r",
            [packed_u8.to(tl.uint32)],
            dtype=(tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _unpack_e2m1_pair_lop3(packed_u8):
        """2-instruction Humming-style bit-shift decode: `e2m1` -> exact FP32 `value * 2^-126`!

        Places sign at bit 31 and `(e1 e0 m0)` at bits 24..22:
          - `0b0000` (0.0) -> `0x00000000` (0.0)
          - `0b0001` (0.5) -> `0x00400000` (`2^-127 = 0.5 * 2^-126`, FP32 subnormal!)
          - `0b0010` (1.0) -> `0x00800000` (`2^-126 = 1.0 * 2^-126`)
          - `0b0111` (6.0) -> `0x01C00000` (`6.0 * 2^-126`)
        Because Triton emits non-FTZ `mul.rn.f32`, multiplying by `e4m3 * 2^+118` is 100% FP32
        bit-exact across all 130,048 finite `(e2m1 x e4m3fn)` pairs (including `+/-0.5` & `-0.0`)!
        """
        u = packed_u8.to(tl.uint32)
        even_bits = ((u & 0x08) << 28) | ((u & 0x07) << 22)
        odd_bits = ((u & 0x80) << 24) | ((u & 0x70) << 18)
        return even_bits.to(tl.float32, bitcast=True), odd_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _decode_e4m3_scaled_2p118(u8_val):
        """Branchless `e4m3fn -> exact value * 2^+118` (max `448 * 2^118 < 2^127`, zero overflow!)."""
        u = u8_val.to(tl.uint32)
        # Normal AND subnormal e4m3fn both map cleanly to `value * 2^-120` with zero branches!
        raw = (((u & 0x80) << 24) | ((u & 0x7F) << 20)).to(tl.float32, bitcast=True)
        # Multiply by 2^120 then 2^118 -> `value * 2^+118` (combined with 2^-126 gives exact 2^-8!)
        return (raw * 1.3292279957849159e36) * 3.3230699894622897e35

    @triton.jit
    def _decode_e4m3fn_u8_to_fp32(u8_val):
        """Branchless software decode `float8_e4m3fn` raw byte -> exact `float32` on SM86."""
        u = u8_val.to(tl.uint32)
        raw = (((u & 0x80) << 24) | ((u & 0x7F) << 20)).to(tl.float32, bitcast=True)
        return raw * 1.3292279957849159e36  # * 2^+120

    @triton.jit
    def _nvfp4_lop3_gemm_kernel(
        X,          # [M, K] bf16
        W_packed,   # [N, K // 2] uint8
        W_scale,    # [N, K // 16] uint8 (raw float8_e4m3fn bytes)
        OuterScale, # [1] float32 = amax / 2688.0
        Out,        # [M, N] bf16
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """Exercise 3A v3: bit-shift `e2m1`/`e4m3` + scale-before-interleave in `bf16` (96 regs!)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        row_mask = rows < M
        col_mask = cols < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x_tile = tl.load(
                X + rows[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            offs_kp = (k0 // 2) + tl.arange(0, BK // 2)
            w_u8 = tl.load(
                W_packed + cols[:, None].to(tl.int64) * (K // 2) + offs_kp[None, :],
                mask=col_mask[:, None],
                other=0,
            )
            offs_ks = (k0 // 16) + tl.arange(0, BK // 16)
            s_u8 = tl.load(
                W_scale + cols[:, None].to(tl.int64) * (K // 16) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            )

            w_even, w_odd = _unpack_e2m1_pair_lop3(w_u8)
            s_f32 = _decode_e4m3_scaled_2p118(s_u8)
            # Both even & odd nibbles share the same per-16 scale -> scale & cast to bf16 BEFORE interleave!
            s_half = tl.reshape(
                tl.broadcast_to(s_f32[:, :, None], (BN, BK // 16, 8)),
                (BN, BK // 2),
            )
            w_even_bf = (w_even * s_half).to(tl.bfloat16)
            w_odd_bf = (w_odd * s_half).to(tl.bfloat16)
            w_bf16 = tl.interleave(w_even_bf, w_odd_bf)
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc)

        # Compensate for 2^-126 (e2m1) * 2^+118 (e4m3 scale) = 2^-8 -> multiply once by 256.0!
        outer = tl.load(OuterScale) * 256.0
        acc = acc * outer
        tl.store(
            Out + rows[:, None].to(tl.int64) * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _nvfp4_w4a16_gemm_kernel(
        X,          # [M, K] bf16
        W_packed,   # [N, K // 2] uint8
        W_scale,    # [N, K // 16] uint8 (raw float8_e4m3fn bytes)
        OuterScale, # [1] float32 = amax / 2688.0
        Out,        # [M, N] bf16
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        row_mask = rows < M
        col_mask = cols < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            # Load X [BM, BK] bf16
            offs_k = k0 + tl.arange(0, BK)
            x_tile = tl.load(
                X + rows[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )

            # Load packed W [BN, BK // 2] uint8 & scales [BN, BK // 16] uint8
            offs_kp = (k0 // 2) + tl.arange(0, BK // 2)
            w_u8 = tl.load(
                W_packed + cols[:, None].to(tl.int64) * (K // 2) + offs_kp[None, :],
                mask=col_mask[:, None],
                other=0,
            )
            offs_ks = (k0 // 16) + tl.arange(0, BK // 16)
            s_u8 = tl.load(
                W_scale + cols[:, None].to(tl.int64) * (K // 16) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            )

            # Fast PTX `prmt.b32` nibble unpack -> [BN, BK]
            w_even, w_odd = _unpack_e2m1_pair_ptx(w_u8)
            w_fp32 = tl.interleave(w_even, w_odd)  # [BN, BK]
            s_fp32 = _decode_e4m3fn_u8_to_fp32(s_u8)
            s_exp = tl.reshape(
                tl.broadcast_to(s_fp32[:, :, None], (BN, BK // 16, 16)),
                (BN, BK),
            )
            w_bf16 = (w_fp32 * s_exp).to(tl.bfloat16)
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc)

        outer = tl.load(OuterScale)
        acc = acc * outer
        tl.store(
            Out + rows[:, None].to(tl.int64) * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _mxfp8_w8a16_gemm_kernel(
        X,        # [M, K] bf16
        W_e4m3,   # [N, K] uint8 (raw float8_e4m3fn bytes)
        W_ue8m0,  # [N, K // 32] uint8
        Out,      # [M, N] bf16
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BM + tl.arange(0, BM)
        cols = pid_n * BN + tl.arange(0, BN)
        row_mask = rows < M
        col_mask = cols < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x_tile = tl.load(
                X + rows[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            w_u8 = tl.load(
                W_e4m3 + cols[:, None].to(tl.int64) * K + offs_k[None, :],
                mask=col_mask[:, None],
                other=0,
            )
            offs_ks = (k0 // 32) + tl.arange(0, BK // 32)
            exp_u8 = tl.load(
                W_ue8m0 + cols[:, None].to(tl.int64) * (K // 32) + offs_ks[None, :],
                mask=col_mask[:, None],
                other=0,
            ).to(tl.uint32)

            scale_bits = tl.where(exp_u8 == 0, 0x00400000, exp_u8 << 23)
            scale_f32 = scale_bits.to(tl.float32, bitcast=True)
            scale_exp = tl.reshape(
                tl.broadcast_to(scale_f32[:, :, None], (BN, BK // 32, 32)),
                (BN, BK),
            )
            w_bf16 = (_decode_e4m3fn_u8_to_fp32(w_u8) * scale_exp).to(tl.bfloat16)
            acc = tl.dot(x_tile, tl.trans(w_bf16), acc)

        tl.store(
            Out + rows[:, None].to(tl.int64) * N + cols[None, :],
            acc.to(tl.bfloat16),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @triton.jit
    def _situ_kernel(
        GateUp,
        Gamma,
        Out,
        N_INTER: tl.constexpr,
        BLOCK: tl.constexpr,
        GATE_CAP: tl.constexpr,
        UP_CAP: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < N_INTER
        gate = tl.load(GateUp + row * (2 * N_INTER) + 2 * cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(GateUp + row * (2 * N_INTER) + 2 * cols + 1, mask=mask, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp2(-gate * 1.4426950408889634))
        g_cap = GATE_CAP * libdevice.tanh(gate / GATE_CAP)
        u_cap = UP_CAP * libdevice.tanh(up / UP_CAP)
        gamma = tl.load(Gamma + row).to(tl.float32)
        y = ((g_cap * sig) * u_cap) * gamma
        tl.store(Out + row * N_INTER + cols, y.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _verify_all_products_kernel(W_bytes, S_bytes, Out):
        """Exhaustively prove `(e2m1 * 2^-126) * (e4m3 * 2^+118) * 256.0` is FP32 bit-exact."""
        pid = tl.program_id(0)  # 0..255 (every possible packed uint8 byte)
        w = tl.load(W_bytes + pid)
        s = tl.load(S_bytes + tl.arange(0, 256))
        even, odd = _unpack_e2m1_pair_lop3(w)
        sf = _decode_e4m3_scaled_2p118(s)
        tl.store(Out + (2 * pid) * 256 + tl.arange(0, 256), (even * sf) * 256.0)
        tl.store(Out + (2 * pid + 1) * 256 + tl.arange(0, 256), (odd * sf) * 256.0)

    def run_nvfp4_gemm(
        x: torch.Tensor,
        w_packed: torch.Tensor,
        w_scale_e4m3: torch.Tensor,
        outer_scale_f32: torch.Tensor,
        *,
        bn: int = 32,
        bk: int | None = None,
        variant: str = "lop3",
    ) -> torch.Tensor:
        m_orig, k = x.shape
        n = w_packed.shape[0]
        bm = 16 if m_orig <= 16 else 32 if m_orig <= 64 else 64
        # SM86 tuning:
        # - BN=32 gives 2304/32 = 72 CTAs (88% of RTX 3090's 82 SMs in 1 clean wave; BN=64 only launches 36!)
        # - BK=256 makes each packed weight row 128 bytes (1 exact GDDR6X cache line) and each scale row 16 bytes!
        if bk is None:
            bk = 256 if (k % 256 == 0) else 128 if (k % 128 == 0) else 64
        out = torch.empty((m_orig, n), dtype=torch.bfloat16, device=x.device)
        grid = (triton.cdiv(m_orig, bm), triton.cdiv(n, bn))
        kernel = _nvfp4_lop3_gemm_kernel if variant == "lop3" else _nvfp4_w4a16_gemm_kernel
        compiled = kernel[grid](
            x,
            w_packed,
            w_scale_e4m3.view(torch.uint8),
            outer_scale_f32,
            out,
            m_orig,
            N=n,
            K=k,
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=4,
            num_stages=3,
        )
        run_nvfp4_gemm.last_compiled = compiled  # type: ignore[attr-defined]
        return out

    run_nvfp4_gemm.verify_all_products_kernel = _verify_all_products_kernel  # type: ignore[attr-defined]

    def run_mxfp8_gemm(
        x: torch.Tensor,
        w_e4m3: torch.Tensor,
        w_ue8m0: torch.Tensor,
    ) -> torch.Tensor:
        m_orig, k = x.shape
        n = w_e4m3.shape[0]
        bm = 16 if m_orig <= 16 else 32 if m_orig <= 64 else 64
        bn = 32
        bk = 128 if (k % 128 == 0) else 64
        out = torch.empty((m_orig, n), dtype=torch.bfloat16, device=x.device)
        grid = (triton.cdiv(m_orig, bm), triton.cdiv(n, bn))
        _mxfp8_w8a16_gemm_kernel[grid](
            x,
            w_e4m3.view(torch.uint8),
            w_ue8m0,
            out,
            m_orig,
            N=n,
            K=k,
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=4,
            num_stages=3,
        )
        return out

    def run_situ(gate_up: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        rows, n2 = gate_up.shape
        n_inter = n2 // 2
        out = torch.empty((rows, n_inter), dtype=torch.bfloat16, device=gate_up.device)
        block = 128
        _situ_kernel[(rows, triton.cdiv(n_inter, block))](
            gate_up,
            gamma,
            out,
            N_INTER=n_inter,
            BLOCK=block,
            GATE_CAP=4.0,
            UP_CAP=25.0,
            num_warps=4,
        )
        return out

    return run_nvfp4_gemm, run_mxfp8_gemm, run_situ


def main() -> None:
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Lab 3: SM86 W4A16 `NVFP4` & W8A16 `MXFP8` + `SiTU` ({device}) ===")

    # 1. Verify our `prmt.b32` byte-LUT table exhaustively across all 256 possible uint8 byte values!
    all_bytes = torch.arange(256, dtype=torch.uint8).unsqueeze(0)
    lut_ref = torch.tensor(E2M1_LUT, dtype=torch.bfloat16)
    expected_pairs = torch.stack(
        (lut_ref[(all_bytes & 0xF).long()], lut_ref[(all_bytes >> 4).long()]), dim=-1
    ).flatten(-2)
    actual_pairs = bitwise_e2m1_to_bf16_cpu_reference(all_bytes)
    # Compare values (note -0.0 == 0.0) and sign bits!
    assert torch.equal(actual_pairs.view(torch.int16), expected_pairs.view(torch.int16))
    print("[Bitwise Proof] All 256 packed `NVFP4` bytes match IEEE `e2m1 -> bf16` bit-for-bit (incl. -0.0)!")

    # 2. Validate against REAL Inkling-2-Mini Expert 0 weights (Layer 0 NVFP4 + Layer 20 MXFP8)
    nvfp4_real = fetch_real_nvfp4_expert_sample(layer_id=0, num_experts=1)
    mxfp8_real = fetch_real_mxfp8_expert_sample(layer_id=20, num_experts=1)

    w13_nv_bf16 = dequant_nvfp4_weight(
        nvfp4_real["w13_weight"][0],
        nvfp4_real["w13_weight_scale"][0],
        nvfp4_real["w13_weight_amax"][0],
        nvfp4_real["w13_denom"],
    ).to(device)
    w2_nv_bf16 = dequant_nvfp4_weight(
        nvfp4_real["w2_weight"][0],
        nvfp4_real["w2_weight_scale"][0],
        nvfp4_real["w2_weight_amax"][0],
        nvfp4_real["w2_denom"],
    ).to(device)
    w13_mx_bf16 = dequant_mxfp8_weight(
        mxfp8_real["w13_weight"][0],
        mxfp8_real["w13_weight_scale"][0],
    ).to(device)

    tokens = 32
    x = (torch.randn(tokens, 2304, device=device, dtype=torch.bfloat16) * 0.5).contiguous()
    gamma = torch.rand(tokens, device=device, dtype=torch.float32) + 0.2

    ref_expert_out = expert_mlp_ref(x, w13_nv_bf16, w2_nv_bf16, gamma)
    ref_mx_out = (x.float() @ w13_mx_bf16.float().T).to(torch.bfloat16)
    print(
        f"[Real Checkpoint Oracle] Layer 0 NVFP4 Expert 0 out std={ref_expert_out.float().std().item():.5f} | "
        f"Layer 20 MXFP8 Expert 0 w13 std={ref_mx_out.float().std().item():.5f}"
    )

    if not torch.cuda.is_available():
        print("\n[CPU Mode] Real Inkling-2-Mini NVFP4 + MXFP8 + SiTU references verified!")
        print("Run `./scripts/remote.sh lab3` on your 3090s to execute PTX `prmt.b32` Tensor Core GEMMs!")
        return

    run_nvfp4_gemm, run_mxfp8_gemm, run_situ = build_sm86_moe_triton_kernels()

    # 2b. Exhaustively verify ALL 130,048 finite (e2m1 x e4m3fn) products on GPU silicon!
    w_all = torch.arange(256, dtype=torch.uint8, device=device)
    s_all = torch.arange(256, dtype=torch.uint8, device=device)
    prod_gpu = torch.empty((512, 256), dtype=torch.float32, device=device)
    run_nvfp4_gemm.verify_all_products_kernel[(256,)](w_all, s_all, prod_gpu)  # type: ignore[attr-defined]
    lut_f32 = torch.tensor(E2M1_LUT, dtype=torch.float32, device=device)
    w_ref = torch.stack((lut_f32[(w_all & 0xF).long()], lut_f32[(w_all >> 4).long()]), dim=-1).flatten()
    s_ref = s_all.view(torch.float8_e4m3fn).float()
    finite_mask = ~torch.isnan(s_ref)  # 254 finite e4m3fn encodings (excludes 0x7F and 0xFF NaNs)
    ref_prod = w_ref[:, None] * s_ref[None, :]
    got_finite = prod_gpu[:, finite_mask]
    ref_finite = ref_prod[:, finite_mask]
    assert torch.equal(got_finite.view(torch.int32), ref_finite.view(torch.int32))
    print(
        f"[Bitwise GPU Proof] All {got_finite.numel():,} finite `(e2m1 x e4m3fn)` products "
        "via `lop3` exponent-shift are 100% FP32 BIT-EXACT (including subnormals & -0.0)!"
    )

    w13_packed = nvfp4_real["w13_weight"][0].to(device).contiguous()
    w13_scale = nvfp4_real["w13_weight_scale"][0].to(device).contiguous()
    w13_outer = (nvfp4_real["w13_weight_amax"][0] / nvfp4_real["w13_denom"]).to(device).float()

    w2_packed = nvfp4_real["w2_weight"][0].to(device).contiguous()
    w2_scale = nvfp4_real["w2_weight_scale"][0].to(device).contiguous()
    w2_outer = (nvfp4_real["w2_weight_amax"][0] / nvfp4_real["w2_denom"]).to(device).float()

    gate_up = run_nvfp4_gemm(x, w13_packed, w13_scale, w13_outer, variant="lop3")
    act = run_situ(gate_up, gamma)
    down = run_nvfp4_gemm(act, w2_packed, w2_scale, w2_outer, variant="lop3")

    gate_up_prmt = run_nvfp4_gemm(x, w13_packed, w13_scale, w13_outer, variant="prmt")
    down_prmt = run_nvfp4_gemm(run_situ(gate_up_prmt, gamma), w2_packed, w2_scale, w2_outer, variant="prmt")

    nv_err = (down.float() - ref_expert_out.float()).abs().max().item()
    nv_prmt_err = (down_prmt.float() - ref_expert_out.float()).abs().max().item()
    situ_ref = situ_activation_ref(gate_up, gamma)
    situ_err = (act.float() - situ_ref.float()).abs().max().item()

    mx_w = mxfp8_real["w13_weight"][0].to(device).contiguous()
    mx_s = mxfp8_real["w13_weight_scale"][0].to(device).contiguous()
    mx_out = run_mxfp8_gemm(x, mx_w, mx_s)
    mx_err = (mx_out.float() - ref_mx_out.float()).abs().max().item()

    print(f"\nSM86 Kernel Parity on Real Inkling-2-Mini Weights (M={tokens}):")
    print(f"  NVFP4 W4A16 v2 (`prmt.b32` LUT) full MLP max|err|        : {nv_prmt_err:.5e}")
    print(f"  NVFP4 W4A16 v3 (`lop3` bit-shift) full MLP max|err|      : {nv_err:.5e}")
    print(f"  SiTU fused activation (`gate_cap=4, up_cap=25`) max|err| : {situ_err:.5e}")
    print(f"  MXFP8 W8A16 (`UE8M0` + Tensor Core) w13 GEMM max|err|    : {mx_err:.5e}")
    assert nv_err < 2e-2 and nv_prmt_err < 2e-2 and situ_err < 1e-2 and mx_err < 5e-2

    # Benchmark W4A16 packed bandwidth savings vs BF16 dense matmul at decode M=8 (top-8 tokens):
    # - L2-hot (single expert) AND L2-cold (rotating 8 distinct experts = 24 MB > 6 MB GA102 L2!)
    x_dec = torch.randn(8, 2304, device=device, dtype=torch.bfloat16) * 0.5

    def bench_graph_hot(fn, graph_iters: int = 300) -> float:
        for _ in range(10):
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
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_ev.record()
        g.replay()
        e_ev.record()
        torch.cuda.synchronize()
        return (s_ev.elapsed_time(e_ev) * 1000.0) / graph_iters

    num_cold = 8
    w_cold = [w13_packed.clone() for _ in range(num_cold)]
    s_cold = [w13_scale.clone() for _ in range(num_cold)]
    o_cold = [w13_outer.clone() for _ in range(num_cold)]
    bf_cold = [w13_nv_bf16.clone() for _ in range(num_cold)]

    def bench_graph_cold(fn_i, rounds: int = 40) -> float:
        for i in range(num_cold):
            fn_i(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(rounds):
                for i in range(num_cold):
                    fn_i(i)
        torch.cuda.synchronize()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_ev.record()
        g.replay()
        e_ev.record()
        torch.cuda.synchronize()
        return (s_ev.elapsed_time(e_ev) * 1000.0) / (rounds * num_cold)

    packed_bytes = w13_packed.numel() + w13_scale.numel()
    bf16_bytes = w13_nv_bf16.numel() * 2
    bf16_hot = bench_graph_hot(lambda: x_dec @ w13_nv_bf16.T)
    bf16_cold = bench_graph_cold(lambda i: x_dec @ bf_cold[i].T)

    v1_hot = bench_graph_hot(
        lambda: run_nvfp4_gemm(x_dec, w13_packed, w13_scale, w13_outer, bn=64, bk=64, variant="prmt")
    )
    v1_c = run_nvfp4_gemm.last_compiled  # type: ignore[attr-defined]
    v1_cold = bench_graph_cold(
        lambda i: run_nvfp4_gemm(x_dec, w_cold[i], s_cold[i], o_cold[i], bn=64, bk=64, variant="prmt")
    )
    v2_hot = bench_graph_hot(
        lambda: run_nvfp4_gemm(x_dec, w13_packed, w13_scale, w13_outer, bn=32, bk=256, variant="prmt")
    )
    v2_c = run_nvfp4_gemm.last_compiled  # type: ignore[attr-defined]
    v2_cold = bench_graph_cold(
        lambda i: run_nvfp4_gemm(x_dec, w_cold[i], s_cold[i], o_cold[i], bn=32, bk=256, variant="prmt")
    )
    v3_hot = bench_graph_hot(
        lambda: run_nvfp4_gemm(x_dec, w13_packed, w13_scale, w13_outer, bn=32, bk=256, variant="lop3")
    )
    v3_c = run_nvfp4_gemm.last_compiled  # type: ignore[attr-defined]
    v3_cold = bench_graph_cold(
        lambda i: run_nvfp4_gemm(x_dec, w_cold[i], s_cold[i], o_cold[i], bn=32, bk=256, variant="lop3")
    )

    print(
        f"\nSingle-Expert `w13` [2304, 2304] @ M=8 Decode (CUDA-Graph GPU; "
        f"{packed_bytes / 1e6:.2f} MB vs {bf16_bytes / 1e6:.2f} MB = 3.56x smaller):"
    )
    print(f"  {'Kernel':<54} | {'L2-Hot':>10} | {'L2-Cold (8E)':>12} | {'Cold Speedup':>12}")
    for label, hot, cold in [
        ("BF16 cuBLAS (dense 10.62 MB)", bf16_hot, bf16_cold),
        (f"Lab 3 v1 prmt (BN=64,BK=64, {v1_c.n_regs:3d} regs, 36 CTAs)", v1_hot, v1_cold),
        (f"Lab 3 v2 prmt (BN=32,BK=256, {v2_c.n_regs:3d} regs, 72 CTAs)", v2_hot, v2_cold),
        (f"Lab 3 v3 lop3 (BN=32,BK=256, {v3_c.n_regs:3d} regs, 72 CTAs)", v3_hot, v3_cold),
    ]:
        print(f"  {label:<54} | {hot:7.2f} us | {cold:9.2f} us | {bf16_cold / cold:11.2f}x")

    try:
        from humming import dtypes as h_dtypes
        from humming.config import WeightScaleType
        from humming.layer import HummingLayer
        from humming.schema import HummingWeightSchema

        schema_nv = HummingWeightSchema(
            b_dtype=h_dtypes.float4e2m1,
            bs_dtype=h_dtypes.float8e4m3,
            weight_scale_group_size=16,
            weight_scale_type=WeightScaleType.GROUP_TENSOR,
        )
        hum_cold = []
        for _ in range(num_cold):
            layer_nv = HummingLayer(2304, 2304, weight_config=schema_nv, torch_dtype=torch.bfloat16).to(
                device
            )
            with torch.no_grad():
                layer_nv.weight.copy_(w13_packed.view(layer_nv.weight.dtype))
                layer_nv.weight_scale.copy_(w13_scale.view(layer_nv.weight_scale.dtype))
                layer_nv.global_scale.copy_(w13_outer.reshape(layer_nv.global_scale.shape))
            layer_nv.transform()
            hum_cold.append(layer_nv)
        hum_hot = bench_graph_hot(lambda: hum_cold[0](x_dec))
        hum_cold_us = bench_graph_cold(lambda i: hum_cold[i](x_dec))
        print(
            f"  {'Humming SM86 C++/PTX (repacked MMA, vLLM prod)':<54} | {hum_hot:7.2f} us | "
            f"{hum_cold_us:9.2f} us | {bf16_cold / hum_cold_us:11.2f}x"
        )
    except Exception:
        pass


if __name__ == "__main__":
    main()
