"""End-to-end verification of Inkling 2 Mini references, real safetensor slices, and SM86 kernels."""

from __future__ import annotations

import pytest
import torch

from inkling_sm86.checkpoint_slice import (
    fetch_real_mxfp8_expert_sample,
    fetch_real_nvfp4_expert_sample,
    fetch_tensor_leading_slice,
)
from inkling_sm86.reference import (
    E2M1_LUT,
    INKLING2_MINI_SPEC,
    dequant_mxfp8_weight,
    dequant_nvfp4_weight,
    estimate_vram_budget,
    expert_mlp_ref,
    loop_blend_rms_norm_ref,
    loop_pair_rms_norm_ref,
    paged_rel_attention_ref,
    shear_rel_logits_ref,
    situ_activation_ref,
)
from labs.lab2_sm86_rel_attention import choose_sm86_decode_splits
from labs.lab3_sm86_nvfp4_mxfp8_moe import bitwise_e2m1_to_bf16_cpu_reference


def test_prmt_b32_e2m1_lut_matches_all_256_bytes_bitwise():
    all_bytes = torch.arange(256, dtype=torch.uint8).unsqueeze(0)
    lut = torch.tensor(E2M1_LUT, dtype=torch.bfloat16)
    expected = torch.stack(
        (lut[(all_bytes & 0xF).long()], lut[(all_bytes >> 4).long()]), dim=-1
    ).flatten(-2)
    actual = bitwise_e2m1_to_bf16_cpu_reference(all_bytes)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_mxfp8_ue8m0_handles_zero_and_normal_exponents():
    w = torch.tensor([[1.0] * 32 + [2.0] * 32], dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0, 127]], dtype=torch.uint8)
    out = dequant_mxfp8_weight(w, scales, out_dtype=torch.float32)
    assert torch.allclose(out[:, :32], torch.full((1, 32), 2.0**-126))
    assert torch.allclose(out[:, 32:], torch.full((1, 32), 2.0))


def test_situ_evaluates_sigmoid_before_gate_softcap():
    # At large positive gate=20.0:
    #   correct Inkling SiTU: (4 * tanh(20/4)) * sigmoid(20) * (25 * tanh(1/25)) ≈ 4.0 * 1.0 * 0.99947 ≈ 4.0
    #   wrong order `silu(softcap(gate))`: 4.0 * sigmoid(4.0) * 1.0 ≈ 3.928 (off by ~1.8%!)
    gate_up = torch.tensor([[20.0, 1.0]], dtype=torch.bfloat16)
    y = situ_activation_ref(gate_up, routing_weight=1.0, gate_cap=4.0, up_cap=25.0).item()
    wrong = (4.0 * torch.sigmoid(torch.tensor(4.0)) * (25.0 * torch.tanh(torch.tensor(1.0 / 25.0)))).item()
    assert abs(y - 4.0) < 0.03
    assert abs(y - wrong) > 0.05


def test_vram_budget_fits_64k_on_tp1_and_128k_on_tp2():
    assert INKLING2_MINI_SPEC.num_layer_visits == 48
    assert INKLING2_MINI_SPEC.num_full_attn_visits == 8
    assert INKLING2_MINI_SPEC.num_swa_attn_visits == 40

    tp1_64k = estimate_vram_budget(65536, tp_size=1)
    tp2_128k = estimate_vram_budget(131072, tp_size=2)
    assert tp1_64k.headroom_on_24gib_card_gib > 2.5
    assert tp2_128k.headroom_on_24gib_card_gib > 10.0


def test_sm86_split_kv_fills_82_sms_for_single_request():
    # 1 request x 8 KV heads = 8 base CTAs -> 10 splits = 80 CTAs across 82 SMs
    assert choose_sm86_decode_splits(num_reqs=1, num_kv_heads=8, max_visible_kv=32768) == 10
    # Large batch (16 reqs x 8 KV heads = 128 CTAs >= 82 SMs) -> no splitting needed
    assert choose_sm86_decode_splits(num_reqs=16, num_kv_heads=8, max_visible_kv=32768) == 1


def test_sheared_bias_alignment_and_attention_reference():
    torch.manual_seed(123)
    seqs = [(1, 1500), (37, 600)]
    total_q = sum(ql for ql, _ in seqs)
    q = torch.randn(total_q, 24, 128, dtype=torch.bfloat16)
    rel_nat = torch.randn(total_q, 24, 1024, dtype=torch.bfloat16)
    sheared = shear_rel_logits_ref(rel_nat, seqs, tile=128, is_local=True)
    assert sheared.shape == (total_q + 128, 24, 1280)

    # Verify decode token 0 (`q_len=1, kv_len=1500 => q_pos=1499`) maps distance 0 to key 1499
    # tile_end = 1, n_max = cdiv(1499 + 1, 128) = 12, key_base = 12*128 - 1280 = 256
    # key 1499 -> column 1499 - 256 = 1243
    assert torch.allclose(sheared[0, :, 1243], rel_nat[0, :, 0])

    k_cache = torch.randn(200, 16, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    bt = torch.arange(200, dtype=torch.int32).view(2, 100)
    gate = torch.randn(total_q, 24 * 128, dtype=torch.bfloat16)
    sinks = torch.randn(total_q, 24, dtype=torch.bfloat16)

    gated_out, lse = paged_rel_attention_ref(
        q, k_cache, v_cache, rel_nat, seqs, bt, window_size=1024, gate=gate, sinks=sinks
    )
    assert gated_out.shape == (total_q, 24, 128)
    assert gated_out.dtype == torch.bfloat16
    assert lse.shape == (24, total_q)
    assert not torch.isnan(gated_out).any()


def test_loop_pair_and_blend_rmsnorm():
    torch.manual_seed(9)
    dest = torch.randn(1, 5, 2304, dtype=torch.float32)
    msg = torch.randn(4, 5, 2304, dtype=torch.float32)
    gf = torch.randn(4, 5, 4608, dtype=torch.float32)
    nw = torch.ones(4, 2304, dtype=torch.bfloat16)

    pair = loop_pair_rms_norm_ref(dest, msg)
    assert pair.shape == (4, 5, 4608) and pair.dtype == torch.bfloat16

    blended = loop_blend_rms_norm_ref(dest, msg, gf, nw)
    assert blended.shape == (4, 5, 2304) and blended.dtype == torch.bfloat16
    # Unit norm_weight => per-row RMS should be ~1.0
    rms = blended.float().square().mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)


def test_real_inkling2_mini_checkpoint_slices_dequantize_cleanly():
    nv = fetch_real_nvfp4_expert_sample(layer_id=0, num_experts=1)
    mx = fetch_real_mxfp8_expert_sample(layer_id=20, num_experts=1)
    rel_proj = fetch_tensor_leading_slice("model.llm.layers.6.attn.rel_logits_proj.proj")

    assert rel_proj.shape == (16, 1024) and rel_proj.dtype == torch.bfloat16
    w13_nv = dequant_nvfp4_weight(
        nv["w13_weight"][0], nv["w13_weight_scale"][0], nv["w13_weight_amax"][0], nv["w13_denom"]
    )
    w2_nv = dequant_nvfp4_weight(
        nv["w2_weight"][0], nv["w2_weight_scale"][0], nv["w2_weight_amax"][0], nv["w2_denom"]
    )
    w13_mx = dequant_mxfp8_weight(mx["w13_weight"][0], mx["w13_weight_scale"][0])

    assert w13_nv.shape == (2304, 2304) and not torch.isnan(w13_nv).any()
    assert w2_nv.shape == (2304, 1152) and not torch.isnan(w2_nv).any()
    assert w13_mx.shape == (2304, 2304) and not torch.isnan(w13_mx).any()

    x = torch.randn(4, 2304, dtype=torch.bfloat16) * 0.5
    out = expert_mlp_ref(x, w13_nv, w2_nv, routing_weight=1.0)
    assert out.shape == (4, 2304) and not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA SM86 hardware tests")
def test_cuda_sm86_labs_end_to_end():
    from labs.lab1_loop_and_convmix import main as run_lab1
    from labs.lab2_sm86_rel_attention import main as run_lab2
    from labs.lab3_sm86_nvfp4_mxfp8_moe import main as run_lab3

    run_lab1()
    run_lab2()
    run_lab3()


def test_lab3b_moe_align_and_batched_kernels():
    from labs.lab3b_sm86_batched_moe import (
        batched_moe_ref,
        build_sm86_batched_moe_triton_kernels,
        choose_moe_block_m,
        load_cuda_moe_extension,
        make_synthetic_routing,
        moe_align_block_size,
    )

    assert choose_moe_block_m(1, 8, 8) == 16
    assert choose_moe_block_m(16, 8, 8) == 16
    assert choose_moe_block_m(64, 8, 8) == 32

    torch.manual_seed(13)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nv = fetch_real_nvfp4_expert_sample(layer_id=0, num_experts=2)
    mx = fetch_real_mxfp8_expert_sample(layer_id=20, num_experts=2)

    w13_nv_bf = dequant_nvfp4_weight(
        nv["w13_weight"], nv["w13_weight_scale"], nv["w13_weight_amax"], nv["w13_denom"]
    ).to(device)
    w2_nv_bf = dequant_nvfp4_weight(
        nv["w2_weight"], nv["w2_weight_scale"], nv["w2_weight_amax"], nv["w2_denom"]
    ).to(device)
    w13_mx_bf = dequant_mxfp8_weight(mx["w13_weight"], mx["w13_weight_scale"]).to(device)
    w2_mx_bf = dequant_mxfp8_weight(mx["w2_weight"], mx["w2_weight_scale"]).to(device)

    x = (torch.randn(4, 2304, device=device, dtype=torch.bfloat16) * 0.5).contiguous()
    ids, wts = make_synthetic_routing(4, num_experts=2, top_k=2, device=device)
    s_ids, e_ids, n_pad = moe_align_block_size(ids, num_experts=2, block_m=16)
    assert int(n_pad.item()) == s_ids.numel() and e_ids.numel() * 16 == s_ids.numel()

    ref_nv = batched_moe_ref(x, w13_nv_bf, w2_nv_bf, ids, wts)
    ref_mx = batched_moe_ref(x, w13_mx_bf, w2_mx_bf, ids, wts)
    assert not torch.isnan(ref_nv).any() and not torch.isnan(ref_mx).any()

    if not torch.cuda.is_available():
        return

    run_nv_tri, run_mx_tri = build_sm86_batched_moe_triton_kernels()
    cu_moe = load_cuda_moe_extension()

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

    tri_nv = run_nv_tri(
        x, nv_w13_p, nv_w13_s, nv_w13_o, nv_w2_p, nv_w2_s, nv_w2_o, wts, s_ids, e_ids, top_k=2
    )
    tri_mx = run_mx_tri(x, mx_w13_w, mx_w13_s, mx_w2_w, mx_w2_s, wts, s_ids, e_ids, top_k=2)

    act = torch.empty((4, 2, 1152), dtype=torch.bfloat16, device=device)
    out = torch.empty((4, 2304), dtype=torch.bfloat16, device=device)
    f32 = torch.zeros((4, 2304), dtype=torch.float32, device=device)

    cu_nv_f = cu_moe.nvfp4_moe_decode_forward(
        x, nv_w13_p, nv_w13_s.view(torch.uint8), nv_w13_o,
        nv_w2_p, nv_w2_s.view(torch.uint8), nv_w2_o,
        ids, wts, act, out.clone(), f32, False,
    )
    cu_nv_a = cu_moe.nvfp4_moe_decode_forward(
        x, nv_w13_p, nv_w13_s.view(torch.uint8), nv_w13_o,
        nv_w2_p, nv_w2_s.view(torch.uint8), nv_w2_o,
        ids, wts, act, out.clone(), f32, True,
    )
    cu_mx_a = cu_moe.mxfp8_moe_decode_forward(
        x, mx_w13_w.view(torch.uint8), mx_w13_s,
        mx_w2_w.view(torch.uint8), mx_w2_s,
        ids, wts, act, out.clone(), f32,
    )

    assert (tri_nv.float() - ref_nv.float()).abs().max().item() < 2e-2
    assert (cu_nv_f.float() - ref_nv.float()).abs().max().item() < 2e-2
    assert (cu_nv_a.float() - ref_nv.float()).abs().max().item() < 2e-2
    assert (tri_mx.float() - ref_mx.float()).abs().max().item() < 2e-2
    assert (cu_mx_a.float() - ref_mx.float()).abs().max().item() < 2e-2
