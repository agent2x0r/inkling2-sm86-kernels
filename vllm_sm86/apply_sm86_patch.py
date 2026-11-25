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
  7. `parser/inkling.py` + `parser/engine/streaming_parser_engine.py`: fixes O(N^2) streaming
     tool-call parsing freeze (`_inkling_arg_value_extender` + coalesced `_feed_args_text` flush)
     that previously stalled the vLLM APIServer event loop for 50–300+ seconds on large tool calls.
  8. Software FP8 (`E4M3FN`) KV cache on Ampere (`< SM89`):
     - `nvidia/ops/__init__.py`: adds `supports_fp8_kv_ptx()` (`has_device_capability(89)`).
     - `nvidia/ops/attention_sm90.py`: replaces SM89+ `cvt.rn.f16x2.e4m3x2` in `_load_fp8_kv`
       with a 5-op SM80/SM86 PTX unpack (`prmt` + `and` + `shr` + `lop3` + non-FTZ `mul.rn.f16x2`
       by `256.0`) that is bit-exact across all 254 finite E4M3FN values (including subnormals).
     - `nvidia/ops/sconv.py`: adds bit-exact IEEE-754 RNE `_fp32_to_e4m3fn_u8` store path in
       `_kv_conv_tile` when `cache.dtype.element_ty == tl.uint8` (since Triton rejects `fp8e4nv`
       pointer dtypes on `< SM89`).
     - `nvidia/attention.py`: allows `("float16", "fp8", "fp8_e4m3")` KV cache on `< SM90`,
       keeps `kv_cache` as `torch.uint8` on `< SM89`, and advertises FP8 support on
       `Inkling2AttentionBackend`.
     - `speculator_config.py`: extends the BF16 DSpark draft KV guard from SM120 to `< SM90`.
     - `v1/worker/gpu/spec_decode/speculator_config.py`: fixes `_resolve_checkpoint_config_class`
       returning base `DSparkSpeculatorConfig` instead of draft-declared `Inkling2DSparkSpeculatorConfig`
       when `method="dspark"` is passed explicitly, so `adapt_draft_model_config` actually runs.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

MARKER = "# [inkling2-sm86-patched]"
FP8_KV_MARKER = "# [inkling2-sm86-fp8-kv]"

FP8_KV_INIT_ADDITION = f"""{FP8_KV_MARKER}
@functools.cache
def supports_fp8_kv_ptx() -> bool:
    \"\"\"Ada/Hopper/Blackwell (SM89+) supports `cvt.rn.f16x2.e4m3x2` and Triton `fp8e4nv`.\"\"\"
    from vllm.platforms import current_platform

    return bool(current_platform.has_device_capability(89))
"""

FP32_TO_E4M3FN_U8_DEF = f"""@triton.jit
def _fp32_to_e4m3fn_u8(x):  {FP8_KV_MARKER}
    \"\"\"Exact IEEE-754 RNE fp32 -> saturating E4M3FN byte (`uint8`) for `< SM89`.\"\"\"
    u = x.to(tl.uint32, bitcast=True)
    sign = (u >> 24) & 0x80
    abs_u = u & 0x7FFFFFFF
    # Normal range (|x| >= 2^-6 = 0x3C800000): clamp to 448.0 (0x43E00000), RNE drop 20 bits.
    clamped = tl.minimum(abs_u, 0x43E00000)
    norm_mag = ((clamped + 0x0007FFFF + ((clamped >> 20) & 1)) - 0x3C000000) >> 20
    # Subnormal range (|x| < 2^-6): adding 2^14 (ULP = 2^-9) gives exact hardware RNE
    # and carries 7.5 * 2^-9 cleanly to 0x08 (min normal 2^-6).
    sub_mag = (
        (abs_u.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True)
        - 0x46800000
    )
    mag = tl.where(abs_u < 0x3C800000, sub_mag, norm_mag)
    mag = tl.where(abs_u > 0x7F800000, 0x7F, mag)
    return (sign | mag).to(tl.uint8)


"""

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

    # 8. Fix O(N^2) streaming tool-call CPU freeze in InklingParser + StreamingParserEngine
    if inkling2_dir.parent.name == "models":
        vllm_pkg_dir = inkling2_dir.parent.parent
        inkling_parser_path = vllm_pkg_dir / "parser" / "inkling.py"
        if inkling_parser_path.is_file():
            ip_src = inkling_parser_path.read_text(encoding="utf-8")
            if MARKER not in ip_src:
                ip_src = _replace_once(
                    ip_src,
                    "from vllm.parser.engine.parser_engine_config import (\n"
                    "    ParserEngineConfig,\n"
                    "    ParserState,\n"
                    "    Transition,\n"
                    ")\n"
                    "from vllm.parser.utils import extract_json_member_object",
                    "from vllm.parser.engine.parser_engine_config import (\n"
                    f"    ArgValueExtender,  {MARKER}\n"
                    "    ParserEngineConfig,\n"
                    "    ParserState,\n"
                    "    Transition,\n"
                    ")\n"
                    "from vllm.parser.utils import extract_json_member_object, scan_json_object_end",
                    "parser/inkling.py:imports",
                )
                extender_def = (
                    f"\n\ndef _inkling_arg_value_extender(raw_args: str) -> ArgValueExtender | None:  {MARKER}\n"
                    '    """O(len(delta)) streaming extender for an open top-level string value in `args`."""\n'
                    '    span = extract_json_member_object(raw_args, "args")\n'
                    "    if not span:\n"
                    "        return None\n"
                    "    if not raw_args.endswith(span) or scan_json_object_end(span, 0) is not None:\n"
                    "        return None\n"
                    "    depth = 0\n"
                    "    in_string = False\n"
                    "    escape = False\n"
                    "    expecting_value = False\n"
                    "    for ch in span:\n"
                    "        if escape:\n"
                    "            escape = False\n"
                    "            continue\n"
                    "        if in_string:\n"
                    '            if ch == "\\\\":\n'
                    "                escape = True\n"
                    '            elif ch == \'"\':\n'
                    "                in_string = False\n"
                    "            continue\n"
                    '        if ch == \'"\':\n'
                    "            in_string = True\n"
                    '        elif ch in "{[":\n'
                    "            depth += 1\n"
                    "            if depth == 1:\n"
                    "                expecting_value = False\n"
                    '        elif ch in "}]":\n'
                    "            depth -= 1\n"
                    "            if depth == 1:\n"
                    "                expecting_value = False\n"
                    '        elif ch == ":" and depth == 1:\n'
                    "            expecting_value = True\n"
                    '        elif ch == "," and depth == 1:\n'
                    "            expecting_value = False\n"
                    "    if not (in_string and depth == 1 and expecting_value):\n"
                    "        return None\n"
                    "\n"
                    "    def extend(delta: str) -> str | None:\n"
                    "        nonlocal escape\n"
                    "        for ch in delta:\n"
                    "            if escape:\n"
                    "                escape = False\n"
                    "                continue\n"
                    '            if ch == "\\\\":\n'
                    "                escape = True\n"
                    '            elif ch == \'"\':\n'
                    "                return None\n"
                    "        return delta\n"
                    "\n"
                    "    return extend\n"
                )
                ip_src = _replace_once(
                    ip_src,
                    "    return span\n\n\n@functools.cache\ndef inkling_config()",
                    f"    return span\n{extender_def}\n\n@functools.cache\ndef inkling_config()",
                    "parser/inkling.py:extender_def",
                )
                ip_src = _replace_once(
                    ip_src,
                    "        arg_converter=_inkling_arg_converter,\n",
                    "        arg_converter=_inkling_arg_converter,\n"
                    "        arg_value_extender=_inkling_arg_value_extender,\n",
                    "parser/inkling.py:config",
                )
                pending_writes[inkling_parser_path] = ip_src

        spe_path = vllm_pkg_dir / "parser" / "engine" / "streaming_parser_engine.py"
        if spe_path.is_file():
            spe_src = spe_path.read_text(encoding="utf-8")
            if MARKER not in spe_src:
                old_feed = (
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
                    "        return events\n"
                    "\n"
                    "    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:\n"
                    "        self._args_buffer += ch\n"
                    "\n"
                    "        if self._args_escape_next:\n"
                    "            self._args_escape_next = False\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return self._flush_safe_args()\n"
                    "\n"
                    "        if self._args_in_string:\n"
                    '            if ch == "\\\\":\n'
                    "                self._args_escape_next = True\n"
                    '            elif ch == \'"\':\n'
                    "                self._args_in_string = False\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return self._flush_safe_args()\n"
                    "\n"
                    '        if ch == \'"\':\n'
                    "            self._args_in_string = True\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return self._flush_safe_args()\n"
                    "\n"
                    '        if ch in ("{", "["):\n'
                    "            self._args_brace_depth += 1\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return self._flush_safe_args()\n"
                    "\n"
                    '        if ch in ("}", "]"):\n'
                    "            if self._args_brace_depth > 0:\n"
                    "                self._args_brace_depth -= 1\n"
                    "            if self._args_brace_depth == 0:\n"
                    "                return []\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return self._flush_safe_args()\n"
                    "\n"
                    "        self._args_safe_end = len(self._args_buffer)\n"
                    "        return self._flush_safe_args()"
                )
                new_feed = (
                    f"    def _feed_args_text(self, text: str) -> list[SemanticEvent]:  {MARKER}\n"
                    '        """Feed text into the JSON argument streaming buffer.\n'
                    "\n"
                    "        Streams argument characters incrementally while holding back\n"
                    "        closing braces/brackets that might change as more input arrives.\n"
                    "        Flushes once per chunk instead of per character to avoid O(N) event\n"
                    "        fan-out on large streaming tool calls.\n"
                    '        """\n'
                    "        for ch in text:\n"
                    "            self._advance_args_char(ch)\n"
                    "        return self._flush_safe_args()\n"
                    "\n"
                    "    def _advance_args_char(self, ch: str) -> None:\n"
                    "        self._args_buffer += ch\n"
                    "\n"
                    "        if self._args_escape_next:\n"
                    "            self._args_escape_next = False\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return\n"
                    "\n"
                    "        if self._args_in_string:\n"
                    '            if ch == "\\\\":\n'
                    "                self._args_escape_next = True\n"
                    '            elif ch == \'"\':\n'
                    "                self._args_in_string = False\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return\n"
                    "\n"
                    '        if ch == \'"\':\n'
                    "            self._args_in_string = True\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return\n"
                    "\n"
                    '        if ch in ("{", "["):\n'
                    "            self._args_brace_depth += 1\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return\n"
                    "\n"
                    '        if ch in ("}", "]"):\n'
                    "            if self._args_brace_depth > 0:\n"
                    "                self._args_brace_depth -= 1\n"
                    "            if self._args_brace_depth == 0:\n"
                    "                return\n"
                    "            self._args_safe_end = len(self._args_buffer)\n"
                    "            return\n"
                    "\n"
                    "        self._args_safe_end = len(self._args_buffer)\n"
                    "\n"
                    "    def _feed_args_char(self, ch: str) -> list[SemanticEvent]:\n"
                    "        self._advance_args_char(ch)\n"
                    "        return self._flush_safe_args()"
                )
                spe_src = _replace_once(
                    spe_src, old_feed, new_feed, "parser/engine/streaming_parser_engine.py:feed"
                )
                pending_writes[spe_path] = spe_src

    # 9. Software FP8 (`E4M3FN`) KV cache on Ampere (`< SM89`)
    # 9a. nvidia/ops/__init__.py: add supports_fp8_kv_ptx()
    init_src_fp8 = pending_writes.get(init_path, init_path.read_text(encoding="utf-8"))
    if FP8_KV_MARKER not in init_src_fp8:
        init_src_fp8 = init_src_fp8.rstrip() + "\n\n\n" + FP8_KV_INIT_ADDITION
        pending_writes[init_path] = init_src_fp8

    # 9b. nvidia/ops/attention_sm90.py: software E4M3x2 -> FP16x2 dequant on `< SM89`
    attn_sm90_path = ops_dir / "attention_sm90.py"
    if attn_sm90_path.is_file():
        a90_src = pending_writes.get(attn_sm90_path, attn_sm90_path.read_text(encoding="utf-8"))
        if FP8_KV_MARKER not in a90_src:
            a90_src = _replace_once(
                a90_src,
                "from vllm.triton_utils import tl, triton\n",
                f"from vllm.triton_utils import tl, triton\nfrom . import supports_fp8_kv_ptx  {FP8_KV_MARKER}\n",
                "ops/attention_sm90.py:import",
            )
            old_load_fp8 = (
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
                "    return tl.interleave(lo, hi).to(dtype)"
            )
            new_load_fp8 = (
                f"if supports_fp8_kv_ptx():  {FP8_KV_MARKER}\n"
                "\n"
                "    @triton.jit\n"
                "    def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):\n"
                "        # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.\n"
                "        half = tl.arange(0, D // 2)\n"
                "        pair = tl.load(\n"
                "            ptr.to(tl.pointer_type(tl.uint16))\n"
                "            + tl.multiple_of(offsets // 2, D // 2)[:, None]\n"
                "            + half[None, :],\n"
                "            mask=mask[:, None],\n"
                "            other=0,\n"
                "        )\n"
                "        lo, hi = tl.inline_asm_elementwise(\n"
                '            "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",\n'
                '            "=h,=h,h",\n'
                "            [pair],\n"
                "            dtype=(tl.float16, tl.float16),\n"
                "            is_pure=True,\n"
                "            pack=1,\n"
                "        )\n"
                "        return tl.interleave(lo, hi).to(dtype)\n"
                "\n"
                "else:\n"
                "\n"
                "    @triton.jit\n"
                "    def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):\n"
                "        # Software SM80/SM86 E4M3FN pair -> FP16x2 unpack (bit-exact across all\n"
                "        # 254 finite values incl. subnormals & +-0.0) via non-FTZ mul.rn.f16x2 * 256.\n"
                "        half = tl.arange(0, D // 2)\n"
                "        pair = tl.load(\n"
                "            ptr.to(tl.pointer_type(tl.uint16))\n"
                "            + tl.multiple_of(offsets // 2, D // 2)[:, None]\n"
                "            + half[None, :],\n"
                "            mask=mask[:, None],\n"
                "            other=0,\n"
                "        )\n"
                "        lo, hi = tl.inline_asm_elementwise(\n"
                '            "{ .reg .b32 p32, s, em, h2, sc; "\n'
                '            "cvt.u32.u16 p32, $2; "\n'
                '            "prmt.b32 s, p32, 0, 0x1707; "\n'
                '            "and.b32 em, s, 0x7F007F00; "\n'
                '            "shr.u32 em, em, 1; "\n'
                '            "lop3.b32 h2, s, em, 0x80008000, 0xEC; "\n'
                '            "mov.b32 sc, 0x5C005C00; "\n'
                '            "mul.rn.f16x2 h2, h2, sc; "\n'
                '            "mov.b32 {$0, $1}, h2; }",\n'
                '            "=h,=h,h",\n'
                "            [pair],\n"
                "            dtype=(tl.float16, tl.float16),\n"
                "            is_pure=True,\n"
                "            pack=1,\n"
                "        )\n"
                "        return tl.interleave(lo, hi).to(dtype)"
            )
            a90_src = _replace_once(
                a90_src, old_load_fp8, new_load_fp8, "ops/attention_sm90.py:_load_fp8_kv"
            )
            pending_writes[attn_sm90_path] = a90_src

    # 9c. nvidia/ops/sconv.py: software fp32 -> E4M3FN uint8 KV store on `< SM89`
    sconv_path = ops_dir / "sconv.py"
    if sconv_path.is_file():
        sc_src = pending_writes.get(sconv_path, sconv_path.read_text(encoding="utf-8"))
        if FP8_KV_MARKER not in sc_src:
            sc_src = _replace_once(
                sc_src,
                "@triton.jit\ndef _kv_conv_tile(",
                f"{FP32_TO_E4M3FN_U8_DEF}@triton.jit\ndef _kv_conv_tile(",
                "ops/sconv.py:_fp32_to_e4m3fn_u8",
            )
            sc_src = _replace_once(
                sc_src,
                "    tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])",
                "    if cache.dtype.element_ty == tl.uint8:\n"
                "        tl.store(dst, _fp32_to_e4m3fn_u8(y), mask=(kv_slot >= 0)[:, None])\n"
                "    else:\n"
                "        tl.store(dst, y.to(cache.dtype.element_ty), mask=(kv_slot >= 0)[:, None])",
                "ops/sconv.py:kv_store",
            )
            pending_writes[sconv_path] = sc_src

    # 9d. nvidia/attention.py: allow FP8/FP16 KV on `< SM90` and keep uint8 storage on `< SM89`
    nv_attn_path = inkling2_dir / "nvidia" / "attention.py"
    if nv_attn_path.is_file():
        na_src = pending_writes.get(nv_attn_path, nv_attn_path.read_text(encoding="utf-8"))
        if FP8_KV_MARKER not in na_src:
            na_src = _replace_once(
                na_src,
                "from .ops import use_sm100_kernels",
                f"from .ops import supports_fp8_kv_ptx, supports_pdl, use_sm100_kernels  {FP8_KV_MARKER}",
                "nvidia/attention.py:import",
            )
            na_src = _replace_once(
                na_src,
                "class Inkling2AttentionBackend(FlashAttentionBackend):\n"
                "    @staticmethod\n"
                "    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:\n"
                "        return Inkling2AttentionMetadataBuilder",
                "class Inkling2AttentionBackend(FlashAttentionBackend):\n"
                "    @staticmethod\n"
                "    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:\n"
                "        return Inkling2AttentionMetadataBuilder\n"
                "\n"
                "    @classmethod\n"
                "    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:\n"
                '        if kv_cache_dtype in ("fp8", "fp8_e4m3"):\n'
                "            return True\n"
                "        return super().supports_kv_cache_dtype(kv_cache_dtype)",
                "nvidia/attention.py:backend_supports_fp8",
            )
            na_src = _replace_once(
                na_src,
                '        supported: tuple[str, ...] = ("auto", "bfloat16")\n'
                "        if current_platform.is_device_capability_family(120):\n"
                '            supported += ("float16", "fp8", "fp8_e4m3")',
                '        supported: tuple[str, ...] = ("auto", "bfloat16")\n'
                "        if current_platform.is_device_capability_family(120) or not supports_pdl():\n"
                '            supported += ("float16", "fp8", "fp8_e4m3")',
                "nvidia/attention.py:supported_kv_dtypes",
            )
            na_src = _replace_once(
                na_src,
                "        if self.fp8_kv:\n"
                "            # Every loop pass owns byte storage containing unit-scale E4M3 KV.\n"
                "            kv_cache = kv_cache.view(torch.float8_e4m3fn)",
                "        if self.fp8_kv and supports_fp8_kv_ptx():\n"
                "            # Every loop pass owns byte storage containing unit-scale E4M3 KV.\n"
                "            kv_cache = kv_cache.view(torch.float8_e4m3fn)",
                "nvidia/attention.py:split_kv_cache",
            )
            pending_writes[nv_attn_path] = na_src

    # 9e. speculator_config.py: keep DSpark draft KV cache in BF16 on `< SM90` as well as SM120
    spec_cfg_path = inkling2_dir / "speculator_config.py"
    if spec_cfg_path.is_file():
        scfg_src = pending_writes.get(spec_cfg_path, spec_cfg_path.read_text(encoding="utf-8"))
        if FP8_KV_MARKER not in scfg_src:
            scfg_src = _replace_once(
                scfg_src,
                "        if current_platform.is_device_capability_family(120):\n"
                "            # The target's clipped, unit-scale FP8 cache does not describe the\n"
                "            # draft's unnormalized values. Keep the draft cache in BF16.\n"
                "            if spec.kv_cache_dtype is None:\n"
                '                spec.kv_cache_dtype = "auto"\n'
                '            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):\n'
                '                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")',
                f"        if (  {FP8_KV_MARKER}\n"
                "            current_platform.is_device_capability_family(120)\n"
                "            or not current_platform.has_device_capability(90)\n"
                "        ):\n"
                "            # The target's clipped, unit-scale FP8 cache does not describe the\n"
                "            # draft's unnormalized values. Keep the draft cache in BF16.\n"
                "            if spec.kv_cache_dtype is None:\n"
                '                spec.kv_cache_dtype = "auto"\n'
                '            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):\n'
                '                raise ValueError("Inkling2 DSpark on SM8x/SM12 requires BF16 draft KV")',
                "speculator_config.py:draft_bf16_kv",
            )
            pending_writes[spec_cfg_path] = scfg_src

    # 9f. v1/worker/gpu/spec_decode/speculator_config.py: return draft-declared subclass
    #     (e.g. Inkling2DSparkSpeculatorConfig) when `method="dspark"` is specified so
    #     its `adapt_draft_model_config` runs and keeps DSpark draft KV in BF16.
    if inkling2_dir.parent.name == "models":
        vllm_pkg_dir = inkling2_dir.parent.parent
        v1_spec_cfg = (
            vllm_pkg_dir / "v1" / "worker" / "gpu" / "spec_decode" / "speculator_config.py"
        )
        if v1_spec_cfg.is_file():
            v1s_src = pending_writes.get(v1_spec_cfg, v1_spec_cfg.read_text(encoding="utf-8"))
            if FP8_KV_MARKER not in v1s_src:
                old_resolve = (
                    "                f\"method={declared!r} if that is what you intended.\"\n"
                    "            )\n"
                    "        return configured_cls\n"
                    "    if spec_config.custom_method is not None:"
                )
                new_resolve = (
                    "                f\"method={declared!r} if that is what you intended.\"\n"
                    "            )\n"
                    f"        if (  {FP8_KV_MARKER}\n"
                    "            spec_config.custom_method is None\n"
                    "            and declared_cls is not None\n"
                    "            and issubclass(declared_cls, configured_cls)\n"
                    "        ):\n"
                    "            return declared_cls\n"
                    "        return configured_cls\n"
                    "    if spec_config.custom_method is not None:"
                )
                v1s_src = _replace_once(
                    v1s_src,
                    old_resolve,
                    new_resolve,
                    "v1/worker/gpu/spec_decode/speculator_config.py:_resolve_checkpoint_config_class",
                )
                pending_writes[v1_spec_cfg] = v1s_src

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
