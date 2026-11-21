#!/usr/bin/env python3
"""Idempotent SM86 (Ampere / Dual RTX 3090) bring-up patcher for vLLM Inkling 2 Mini.

Works against BOTH:
  1. A git checkout of vLLM PR #69028 (`python vllm_sm86/apply_sm86_patch.py /path/to/vllm`)
  2. The installed `vllm` package inside Docker image `vllm/vllm-openai:inkling-2-mini`
     (`python vllm_sm86/apply_sm86_patch.py --installed`) — ZERO C++ rebuild required!

Exact hardware blockers patched:
  1. `quant_config.py`: `get_min_capability() -> 80` (was `90`).
  2. `ops/__init__.py` + 9 `ops/*.py` files: Hopper Grid Dependency Control (`launch_pdl=True`,
     `tl.extra.cuda.gdc_wait()`, `tl.extra.cuda.gdc_launch_dependents()`) emits `.target sm_90`
     PTX (`griddepcontrol.*`) that fails PTX assembly on SM86. Gated via `tl.constexpr` so
     Triton's AST dead-code eliminator strips it cleanly on SM8x while keeping it on SM90+.
  3. `ops/linear.py`: `TMA_MIN_TOKENS` check excluded `sm120` only; extended to exclude `< SM90`
     so Ampere uses `_k_group_kernel` instead of Hopper TMA `TensorDescriptor`.
  4. `ops/conv_mix.py`: `csrc/inkling2/conv_mix.cu` uses Hopper `cp.async.bulk` TMA; routes `< SM90`
     to the existing `short_conv` + `attn_res` Triton path (same path `SM120` uses).
  5. `nvidia/vision.py`: selects `fa_version = 2` on `< SM90` instead of Hopper-only FA3.
  6. `tests/models/inkling2/test_{moe,attention}.py` (if present): unskips SM80/SM86 in pytest.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

MARKER = "# [inkling2-sm86-patched]"

OPS_INIT_ADDITION = f"""{MARKER}
@functools.cache
def supports_pdl() -> bool:
    \"\"\"Hopper+ (SM90+) supports Programmatic Dependent Launch (`griddepcontrol` PTX) and TMA.\"\"\"
    from vllm.platforms import current_platform

    return bool(current_platform.has_device_capability(90))


if supports_pdl():
    from vllm.triton_utils import tl as _tl
    from vllm.triton_utils import triton as _triton

    @_triton.jit
    def gdc_wait():
        _tl.extra.cuda.gdc_wait()

    @_triton.jit
    def gdc_launch_dependents():
        _tl.extra.cuda.gdc_launch_dependents()
else:
    from vllm.triton_utils import triton as _triton

    @_triton.jit
    def gdc_wait():
        return

    @_triton.jit
    def gdc_launch_dependents():
        return
"""

PDL_OPS_FILES: tuple[str, ...] = (
    "activation.py",
    "attention.py",
    "attn_res.py",
    "linear.py",
    "moe.py",
    "moe_sm120.py",
    "mxfp8.py",
    "nvfp4.py",
    "sconv.py",
)


def _replace_once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"[{label}] Expected anchor exactly once, found {count}:\n{old}")
    return src.replace(old, new, 1)


def resolve_inkling2_dirs(target: str) -> tuple[Path, Path | None]:
    """Return `(vllm_models_inkling2_dir, optional_tests_inkling2_dir)`."""
    if target == "--installed":
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("Could not locate installed `vllm` package in current Python env")
        vllm_pkg = Path(list(spec.submodule_search_locations)[0])
        return vllm_pkg / "models" / "inkling2", None

    root = Path(target).resolve()
    if (root / "vllm" / "models" / "inkling2").is_dir():
        tests_dir = root / "tests" / "models" / "inkling2"
        return root / "vllm" / "models" / "inkling2", (tests_dir if tests_dir.is_dir() else None)
    if (root / "models" / "inkling2").is_dir():
        return root / "models" / "inkling2", None
    if (root / "quant_config.py").is_file() and (root / "nvidia").is_dir():
        return root, None
    raise FileNotFoundError(f"Could not find `vllm/models/inkling2` under {root}")


def apply_patches(inkling2_dir: Path, tests_dir: Path | None = None, dry_run: bool = False) -> list[str]:
    touched: list[str] = []
    pending_writes: dict[Path, str] = {}

    # 1. quant_config.py
    qc_path = inkling2_dir / "quant_config.py"
    qc_src = qc_path.read_text(encoding="utf-8")
    if MARKER not in qc_src:
        qc_src = _replace_once(
            qc_src,
            "    def get_min_capability(cls) -> int:\n        return 90",
            f"    def get_min_capability(cls) -> int:  {MARKER}\n        return 80",
            "quant_config.py",
        )
        pending_writes[qc_path] = qc_src

    # 2. nvidia/ops/__init__.py
    ops_dir = inkling2_dir / "nvidia" / "ops"
    init_path = ops_dir / "__init__.py"
    init_src = init_path.read_text(encoding="utf-8")
    if MARKER not in init_src:
        init_src = init_src.rstrip() + "\n\n\n" + OPS_INIT_ADDITION
        pending_writes[init_path] = init_src

    # 3. PDL / GDC replacements across ops/*.py
    for fname in PDL_OPS_FILES:
        fpath = ops_dir / fname
        if not fpath.is_file():
            continue
        src = pending_writes.get(fpath, fpath.read_text(encoding="utf-8"))
        if MARKER in src:
            continue
        needs_gdc_wait = "tl.extra.cuda.gdc_wait()" in src
        needs_gdc_dep = "tl.extra.cuda.gdc_launch_dependents()" in src
        needs_pdl = "launch_pdl=True" in src
        if not (needs_gdc_wait or needs_gdc_dep or needs_pdl):
            continue

        imports = ["supports_pdl"]
        if needs_gdc_wait:
            imports.append("gdc_wait")
            src = src.replace("tl.extra.cuda.gdc_wait()", "gdc_wait()")
        if needs_gdc_dep:
            imports.append("gdc_launch_dependents")
            src = src.replace("tl.extra.cuda.gdc_launch_dependents()", "gdc_launch_dependents()")
        if needs_pdl:
            src = src.replace("launch_pdl=True", "launch_pdl=supports_pdl()")

        import_stmt = f"from . import {', '.join(imports)}  {MARKER}\n"
        # Insert right after the `vllm.triton_utils` import line
        lines = src.splitlines(keepends=True)
        inserted = False
        for idx, line in enumerate(lines):
            if "from vllm.triton_utils import" in line:
                lines.insert(idx + 1, import_stmt)
                inserted = True
                break
        if not inserted:
            raise RuntimeError(f"[{fname}] Could not find `from vllm.triton_utils import` anchor")
        pending_writes[fpath] = "".join(lines)

    # 4. ops/linear.py TMA guard
    lin_path = ops_dir / "linear.py"
    lin_src = pending_writes.get(lin_path, lin_path.read_text(encoding="utf-8"))
    old_tma = "    if tokens >= TMA_MIN_TOKENS and not sm120:"
    new_tma = "    if tokens >= TMA_MIN_TOKENS and not sm120 and supports_pdl():"
    if old_tma in lin_src:
        lin_src = _replace_once(lin_src, old_tma, new_tma, "ops/linear.py:TMA")
        pending_writes[lin_path] = lin_src

    # 5. ops/conv_mix.py SM86 fallback to Triton short_conv + attn_res
    cm_path = ops_dir / "conv_mix.py"
    cm_src = cm_path.read_text(encoding="utf-8")
    if MARKER not in cm_src:
        cm_src = _replace_once(
            cm_src,
            "from vllm.platforms import current_platform",
            f"from vllm.platforms import current_platform\nfrom . import supports_pdl  {MARKER}",
            "ops/conv_mix.py:import",
        )
        cm_src = _replace_once(
            cm_src,
            "        if current_platform.is_device_capability_family(120):",
            "        if current_platform.is_device_capability_family(120) or not supports_pdl():",
            "ops/conv_mix.py:branch",
        )
        pending_writes[cm_path] = cm_src

    # 6. nvidia/vision.py FlashAttention-2 on Ampere
    vis_path = inkling2_dir / "nvidia" / "vision.py"
    vis_src = vis_path.read_text(encoding="utf-8")
    if MARKER not in vis_src:
        vis_src = _replace_once(
            vis_src,
            "from vllm.models.inkling2.nvidia.ops import use_sm100_kernels",
            f"from vllm.models.inkling2.nvidia.ops import supports_pdl, use_sm100_kernels  {MARKER}",
            "nvidia/vision.py:import",
        )
        vis_src = _replace_once(
            vis_src,
            "            version = 2 if current_platform.is_device_capability_family(120) else 3",
            "            version = 2 if (current_platform.is_device_capability_family(120) or not supports_pdl()) else 3",
            "nvidia/vision.py:fa_version",
        )
        pending_writes[vis_path] = vis_src

    # 7. Optional upstream pytest capability guards
    if tests_dir is not None:
        t_moe = tests_dir / "test_moe.py"
        if t_moe.is_file():
            s = t_moe.read_text(encoding="utf-8")
            if MARKER not in s:
                s = _replace_once(
                    s,
                    "    or current_platform.is_device_capability(90)",
                    f"    or current_platform.is_device_capability(90)\n    or current_platform.is_device_capability_family(80)  {MARKER}",
                    "tests/test_moe.py",
                )
                pending_writes[t_moe] = s

        t_attn = tests_dir / "test_attention.py"
        if t_attn.is_file():
            s = t_attn.read_text(encoding="utf-8")
            if MARKER not in s:
                s = _replace_once(
                    s,
                    "    _cap is None or _cap.major not in (9, 10, 11, 12),",
                    f"    _cap is None or _cap.major not in (8, 9, 10, 11, 12),  {MARKER}",
                    "tests/test_attention.py",
                )
                pending_writes[t_attn] = s

    # Validate every patched file parses as valid Python AST BEFORE writing anything!
    for path, content in pending_writes.items():
        ast.parse(content, filename=str(path))

    for path, content in sorted(pending_writes.items()):
        if not dry_run:
            path.write_text(content, encoding="utf-8")
        touched.append(str(path))

    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Path to vLLM checkout or `vllm/models/inkling2` dir (omit with `--installed`)",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="Patch the installed `vllm` package in the active Python environment",
    )
    parser.add_argument("--dry-run", action="store_true", help="Verify anchors & AST without writing")
    args = parser.parse_args(argv)

    if args.installed:
        target = "--installed"
    elif args.target is not None:
        target = args.target
    else:
        parser.error("Provide a target path or pass --installed")

    inkling2_dir, tests_dir = resolve_inkling2_dirs(target)
    touched = apply_patches(inkling2_dir, tests_dir, dry_run=args.dry_run)
    mode = "Verified (dry-run)" if args.dry_run else "Patched"
    if not touched:
        print(f"[OK] Already SM86-patched: {inkling2_dir}")
    else:
        print(f"[OK] {mode} {len(touched)} files under {inkling2_dir}:")
        for p in touched:
            print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
