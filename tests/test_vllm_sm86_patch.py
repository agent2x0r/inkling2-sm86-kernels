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


def test_patcher_cli_dry_run_and_flags(tmp_path: Path):
    ink, _ = _populate_synthetic_vllm_tree(tmp_path / "cli")
    assert main([str(ink), "--dry-run"]) == 0
    assert "return 90" in (ink / "quant_config.py").read_text(encoding="utf-8")
    assert main([str(ink)]) == 0
    assert "return 80" in (ink / "quant_config.py").read_text(encoding="utf-8")
