"""Ampere (SM86 / Dual RTX 3090) kernel lab and reference suite for Inkling 2 Mini."""

from .reference import (
    E2M1_LUT,
    INKLING2_MINI_SPEC,
    VRAMBudget,
    dequant_mxfp8_weight,
    dequant_nvfp4_weight,
    estimate_vram_budget,
    expert_mlp_ref,
    loop_blend_rms_norm_ref,
    loop_pair_rms_norm_ref,
    paged_rel_attention_ref,
    shear_rel_logits_ref,
    shift_rel_proj_ref,
    situ_activation_ref,
)

__all__ = [
    "E2M1_LUT",
    "INKLING2_MINI_SPEC",
    "VRAMBudget",
    "dequant_mxfp8_weight",
    "dequant_nvfp4_weight",
    "estimate_vram_budget",
    "expert_mlp_ref",
    "loop_blend_rms_norm_ref",
    "loop_pair_rms_norm_ref",
    "paged_rel_attention_ref",
    "shear_rel_logits_ref",
    "shift_rel_proj_ref",
    "situ_activation_ref",
]
