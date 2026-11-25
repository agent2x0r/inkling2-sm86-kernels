"""Verify `vllm_sm86/apply_sm86_patch.py` cleanly patches vLLM PR #69028 and is idempotent."""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

from vllm_sm86.apply_sm86_patch import apply_patches, main


def _populate_synthetic_vllm_tree(root: Path) -> tuple[Path, Path]:
    ink = root / "vllm" / "models" / "inkling2"
    ops = ink / "nvidia" / "ops"
    tests = root / "tests" / "models" / "inkling2"
    ops.mkdir(parents=True)
    tests.mkdir(parents=True)

    (ink / "quant_config.py").write_text(
        "class Inkling2QuantConfig:\n"
        "    @classmethod\n"
        "    def get_min_capability(cls) -> int:\n"
        "        return 90\n",
        encoding="utf-8",
    )
    (ops / "__init__.py").write_text(
        "import functools\n\n@functools.cache\ndef use_sm100_kernels() -> bool:\n    return False\n",
        encoding="utf-8",
    )
    (ops / "attention.py").write_text(
        "from vllm.triton_utils import gl, gluon, tl, tldevice, triton\n\n"
        "def f():\n    tl.extra.cuda.gdc_wait()\n    k[1](launch_pdl=True)\n",
        encoding="utf-8",
    )
    (ops / "linear.py").write_text(
        "from vllm.triton_utils import tl, triton\n\n"
        "def g(tokens, sm120, TMA_MIN_TOKENS):\n"
        "    tl.extra.cuda.gdc_wait()\n"
        "    tl.extra.cuda.gdc_launch_dependents()\n"
        "    if tokens >= TMA_MIN_TOKENS and not sm120:\n"
        "        return 1\n"
        "    k[1](launch_pdl=True)\n",
        encoding="utf-8",
    )
    (ops / "conv_mix.py").write_text(
        "from vllm.platforms import current_platform\n\n"
        "def conv_mix():\n"
        "    if True:\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return 1\n",
        encoding="utf-8",
    )
    (ink / "nvidia" / "vision.py").write_text(
        "from vllm.models.inkling2.nvidia.ops import use_sm100_kernels\n\n"
        "def v(current_platform):\n"
        "        if False:\n"
        "            pass\n"
        "        else:\n"
        "            version = 2 if current_platform.is_device_capability_family(120) else 3\n",
        encoding="utf-8",
    )
    (ops / "attention_sm90.py").write_text(
        "import torch\n\n"
        "from vllm.triton_utils import tl, triton\n\n"
        "@triton.jit\n"
        "def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):\n"
        "    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.\n"
        "    half = tl.arange(0, D // 2)\n"
        "    pair = tl.load(\n"
        "        ptr.to(tl.pointer_type(tl.uint16))\n"
        "        + tl.multiple_of(offsets // 2, D // 2)[:, None]\n"
        "        + half[None, :],\n"
        "        mask=mask[:, None],\n"
        "        other=0,\n"
        "    )\n"
        "    lo, hi = tl.inline_asm_elementwise(\n"
        '        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",\n'
        '        "=h,=h,h",\n'
        "        [pair],\n"
        "        dtype=(tl.float16, tl.float16),\n"
        "        is_pure=True,\n"
        "        pack=1,\n"
        "    )\n"
        "    return tl.interleave(lo, hi).to(dtype)\n",
        encoding="utf-8",
    )
    (ops / "sconv.py").write_text(
        "from vllm.triton_utils import tl, triton\n\n"
        "@triton.jit\n"
        "def _kv_conv_tile(dst, y, cache, kv_slot):\n"
        "    tl.extra.cuda.gdc_wait()\n"
        "    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])\n",
        encoding="utf-8",
    )
    (ink / "nvidia" / "attention.py").write_text(
        "from .ops import use_sm100_kernels\n\n"
        "class Inkling2AttentionBackend(FlashAttentionBackend):\n"
        "    @staticmethod\n"
        "    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:\n"
        "        return Inkling2AttentionMetadataBuilder\n\n"
        "class Inkling2Attention:\n"
        "    def __init__(self, vllm_config):\n"
        '        supported: tuple[str, ...] = ("auto", "bfloat16")\n'
        "        if current_platform.is_device_capability_family(120):\n"
        '            supported += ("float16", "fp8", "fp8_e4m3")\n'
        "    def _split_kv_cache(self, kv_cache):\n"
        "        if self.fp8_kv:\n"
        "            # Every loop pass owns byte storage containing unit-scale E4M3 KV.\n"
        "            kv_cache = kv_cache.view(torch.float8_e4m3fn)\n"
        "        return kv_cache\n",
        encoding="utf-8",
    )
    (ink / "speculator_config.py").write_text(
        "class Inkling2DSparkSpeculatorConfig:\n"
        "    @classmethod\n"
        '    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:\n'
        "        super().adapt_draft_model_config(spec)\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            # The target's clipped, unit-scale FP8 cache does not describe the\n"
        "            # draft's unnormalized values. Keep the draft cache in BF16.\n"
        "            if spec.kv_cache_dtype is None:\n"
        '                spec.kv_cache_dtype = "auto"\n'
        '            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):\n'
        '                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")\n',
        encoding="utf-8",
    )
    (tests / "test_moe.py").write_text(
        "if not (\n    current_platform.is_device_capability_family(100)\n"
        "    or current_platform.is_device_capability(90)\n):\n    pass\n",
        encoding="utf-8",
    )
    (tests / "test_attention.py").write_text(
        "requires_sm90 = skipif(\n"
        "    _cap is None or _cap.major not in (9, 10, 11, 12),\n)\n",
        encoding="utf-8",
    )
    parser_dir = root / "vllm" / "parser"
    engine_dir = parser_dir / "engine"
    engine_dir.mkdir(parents=True)
    v1_spec_dir = root / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    v1_spec_dir.mkdir(parents=True)
    (v1_spec_dir / "speculator_config.py").write_text(
        "def _resolve_checkpoint_config_class(spec_config, configured_cls):\n"
        "    declared_cls = None\n"
        "    if spec_config.method is not None:\n"
        "        if declared_cls is not None and declared_cls.base_method != spec_config.method:\n"
        "            raise ValueError(\n"
        '                f"method={declared!r} if that is what you intended."\n'
        "            )\n"
        "        return configured_cls\n"
        "    if spec_config.custom_method is not None:\n"
        "        return configured_cls\n",
        encoding="utf-8",
    )
    (parser_dir / "inkling.py").write_text(
        "import functools\n"
        "from vllm.parser.engine.parser_engine_config import (\n"
        "    ParserEngineConfig,\n"
        "    ParserState,\n"
        "    Transition,\n"
        ")\n"
        "from vllm.parser.utils import extract_json_member_object\n\n"
        "def _inkling_arg_converter(raw_args: str, partial: bool) -> str:\n"
        "    span = extract_json_member_object(raw_args, 'args')\n"
        "    return span\n\n\n"
        "@functools.cache\n"
        "def inkling_config():\n"
        "    return dict(\n"
        "        arg_converter=_inkling_arg_converter,\n"
        "    )\n",
        encoding="utf-8",
    )
    (engine_dir / "streaming_parser_engine.py").write_text(
        "class StreamingParserEngine:\n"
        "    def _feed_args_text(self, text: str) -> list[SemanticEvent]:\n"
        '        """Feed text into the JSON argument streaming buffer.\n'
        "\n"
        "        Streams argument characters incrementally while holding back\n"
        "        closing braces/brackets that might change as more input arrives.\n"
        '        """\n'
        "        events: list[SemanticEvent] = []\n"
        "        for ch in text:\n"
        "            result = self._feed_args_char(ch)\n"
        "            events.extend(result)\n"
        "        return events\n\n"
        "    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:\n"
        "        self._args_buffer += ch\n\n"
        "        if self._args_escape_next:\n"
        "            self._args_escape_next = False\n"
        "            self._args_safe_end = len(self._args_buffer)\n"
        "            return self._flush_safe_args()\n\n"
        "        if self._args_in_string:\n"
        '            if ch == "\\\\":\n'
        "                self._args_escape_next = True\n"
        '            elif ch == \'"\':\n'
        "                self._args_in_string = False\n"
        "            self._args_safe_end = len(self._args_buffer)\n"
        "            return self._flush_safe_args()\n\n"
        '        if ch == \'"\':\n'
        "            self._args_in_string = True\n"
        "            self._args_safe_end = len(self._args_buffer)\n"
        "            return self._flush_safe_args()\n\n"
        '        if ch in ("{", "["):\n'
        "            self._args_brace_depth += 1\n"
        "            self._args_safe_end = len(self._args_buffer)\n"
        "            return self._flush_safe_args()\n\n"
        '        if ch in ("}", "]"):\n'
        "            if self._args_brace_depth > 0:\n"
        "                self._args_brace_depth -= 1\n"
        "            if self._args_brace_depth == 0:\n"
        "                return []\n"
        "            self._args_safe_end = len(self._args_buffer)\n"
        "            return self._flush_safe_args()\n\n"
        "        self._args_safe_end = len(self._args_buffer)\n"
        "        return self._flush_safe_args()\n",
        encoding="utf-8",
    )
    return ink, tests


def test_patcher_against_synthetic_and_real_pr69028_tree(tmp_path: Path):
    real_pr_src = Path("/tmp/inkling2_src/vllm/models/inkling2")
    real_pr_tests = Path("/tmp/inkling2_tests")

    if real_pr_src.is_dir():
        ink = tmp_path / "real" / "vllm" / "models" / "inkling2"
        shutil.copytree(real_pr_src, ink)
        tests = None
        if real_pr_tests.is_dir():
            tests = tmp_path / "real" / "tests" / "models" / "inkling2"
            shutil.copytree(real_pr_tests, tests)
    else:
        ink, tests = _populate_synthetic_vllm_tree(tmp_path / "syn")

    touched_first = apply_patches(ink, tests, dry_run=False)
    assert len(touched_first) >= 6

    # Second pass MUST be a clean no-op (idempotent)
    touched_second = apply_patches(ink, tests, dry_run=False)
    assert touched_second == []

    # Verify zero unguarded SM90 PDL calls remain outside ops/__init__.py
    for py_file in ink.rglob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        ast.parse(src, filename=str(py_file))
        if py_file.name != "__init__.py":
            assert "tl.extra.cuda.gdc_wait()" not in src, f"Unguarded gdc_wait in {py_file}"
            assert "tl.extra.cuda.gdc_launch_dependents()" not in src, f"Unguarded gdc_dep in {py_file}"
            assert "launch_pdl=True" not in src, f"Unguarded launch_pdl=True in {py_file}"

    assert "return 80" in (ink / "quant_config.py").read_text(encoding="utf-8")
    init_src = (ink / "nvidia" / "ops" / "__init__.py").read_text(encoding="utf-8")
    assert "current_platform.has_device_capability(90)" in init_src
    assert "current_platform.has_device_capability(89)" in init_src
    a90_src = (ink / "nvidia" / "ops" / "attention_sm90.py").read_text(encoding="utf-8")
    assert "mul.rn.f16x2 h2, h2, sc;" in a90_src
    assert "if supports_fp8_kv_ptx():" in a90_src
    sconv_src = (ink / "nvidia" / "ops" / "sconv.py").read_text(encoding="utf-8")
    assert "def _fp32_to_e4m3fn_u8(x):" in sconv_src
    assert "if cache.dtype.element_ty == tl.uint8:" in sconv_src
    nv_attn_src = (ink / "nvidia" / "attention.py").read_text(encoding="utf-8")
    assert "if self.fp8_kv and supports_fp8_kv_ptx():" in nv_attn_src
    assert "or not supports_pdl():" in nv_attn_src
    spec_src = (ink / "speculator_config.py").read_text(encoding="utf-8")
    assert "not current_platform.has_device_capability(90)" in spec_src
    v1_spec = (
        ink.parent.parent / "v1" / "worker" / "gpu" / "spec_decode" / "speculator_config.py"
    )
    if v1_spec.is_file():
        assert "issubclass(declared_cls, configured_cls)" in v1_spec.read_text(encoding="utf-8")
    parser_inkling = ink.parent.parent / "parser" / "inkling.py"
    if parser_inkling.is_file():
        assert "arg_value_extender=_inkling_arg_value_extender" in parser_inkling.read_text(encoding="utf-8")
    spe = ink.parent.parent / "parser" / "engine" / "streaming_parser_engine.py"
    if spe.is_file():
        assert "def _advance_args_char" in spe.read_text(encoding="utf-8")


def test_patcher_cli_dry_run_and_flags(tmp_path: Path):
    ink, _ = _populate_synthetic_vllm_tree(tmp_path / "cli")
    assert main([str(ink), "--dry-run"]) == 0
    assert "return 90" in (ink / "quant_config.py").read_text(encoding="utf-8")
    assert main([str(ink)]) == 0
    assert "return 80" in (ink / "quant_config.py").read_text(encoding="utf-8")


def test_software_e4m3fn_store_and_load_bit_exact_on_cpu():
    """Verify the exact bit-level algorithms used in `_fp32_to_e4m3fn_u8` and `_load_fp8_kv`."""
    import torch

    # 1. Software FP32 -> E4M3FN (uint8) RNE conversion vs PyTorch float8_e4m3fn
    specials = torch.tensor(
        [
            0.0, -0.0, 1e-40, -1e-40, 2**-11, 2**-10, -2**-10,
            *[k * 2**-10 for k in range(32)],
            *[-k * 2**-10 for k in range(32)],
            0.015625, 1.0, 1.0625, 1.125, 1.1875, 1.25, -1.0625,
            22.627417, -22.627417, 400.0, 416.0, 432.0, 447.9, 448.0, 448.1, 500.0, -500.0,
            float("inf"), float("-inf"), float("nan"), -float("nan"),
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(7)
    xs = torch.cat([specials, torch.randn(4096) * 0.02, torch.randn(4096) * 20.0, torch.randn(4096) * 300.0])
    u = xs.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = (u >> 24) & 0x80
    abs_u = u & 0x7FFFFFFF
    clamped = torch.minimum(abs_u, torch.tensor(0x43E00000, dtype=torch.int64))
    norm_mag = ((clamped + 0x0007FFFF + ((clamped >> 20) & 1)) - 0x3C000000) >> 20
    abs_f = abs_u.to(torch.int32).view(torch.float32)
    sub_mag = ((abs_f + 16384.0).view(torch.int32).to(torch.int64) & 0xFFFFFFFF) - 0x46800000
    mag = torch.where(abs_u < 0x3C800000, sub_mag, norm_mag)
    mag = torch.where(abs_u > 0x7F800000, torch.tensor(0x7F, dtype=torch.int64), mag)
    sw_u8 = (sign | mag).to(torch.uint8)
    ref_u8 = xs.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(sw_u8, ref_u8)

    # 2. Software E4M3FN (uint8) -> FP16x2 (non-FTZ * 256.0) -> BF16 across all 254 finite bytes
    all_u8 = torch.arange(256, dtype=torch.uint8)
    u16 = all_u8.to(torch.int32)
    s = u16 << 8
    em = (s & 0x7F00) >> 1
    h_raw = ((s & 0x8000) | em).to(torch.int16).view(torch.float16)
    h_val = (h_raw * torch.tensor(256.0, dtype=torch.float16)).to(torch.bfloat16)
    ref_bf16 = all_u8.view(torch.float8_e4m3fn).to(torch.bfloat16)
    finite = (all_u8 != 0x7F) & (all_u8 != 0xFF)
    assert torch.equal(h_val.view(torch.uint16)[finite], ref_bf16.view(torch.uint16)[finite])
