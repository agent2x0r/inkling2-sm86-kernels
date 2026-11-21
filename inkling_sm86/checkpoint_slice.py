"""Stream exact byte slices from official `thinkingmachines/Inkling-2-Mini` via HTTP Range.

Why this exists:
The official checkpoint is a single 18.18 GiB file (`model-00001-of-00001.safetensors`).
During kernel development—especially on a host with 24 GB system RAM or a laptop—you do not
need to load 18+ GiB (or materialize a 52 GiB BF16 copy) just to validate NVFP4/MXFP8
dequantization, SiTU activations, or sheared relative-position attention against real weights.

Safetensors stores a compact JSON header at bytes [8, 8 + header_len) with exact byte offsets
for all 1,244 tensors. Using HTTP `Range` requests, we can pull individual real experts or
attention projections (~30 KB to ~5 MB) in ~1 second and cache them locally under `.cache/slices/`.
"""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.request
from pathlib import Path
from typing import Any

import torch

HF_SAFETENSORS_URL = (
    "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/"
    "model-00001-of-00001.safetensors"
)

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

_DTYPE_VIEW: dict[str, tuple[torch.dtype, int]] = {
    "F32": (torch.float32, 4),
    "BF16": (torch.bfloat16, 2),
    "F16": (torch.float16, 2),
    "U8": (torch.uint8, 1),
    "I8": (torch.int8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
}


def _http_range_get(url: str, start: int, end_inclusive: int, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end_inclusive}",
            "User-Agent": "inkling2-sm86-kernels/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_safetensors_header(
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    url: str = HF_SAFETENSORS_URL,
) -> dict[str, Any]:
    """Fetch and cache the 148 KB JSON header of `model-00001-of-00001.safetensors`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    header_path = cache_dir / "inkling2_mini_safetensors_header.json"
    if header_path.exists():
        return json.loads(header_path.read_text(encoding="utf-8"))

    raw_len = _http_range_get(url, 0, 7)
    header_len = struct.unpack("<Q", raw_len)[0]
    raw_hdr = _http_range_get(url, 8, 8 + header_len - 1)
    tensors = json.loads(raw_hdr.decode("utf-8"))
    payload = {"header_len": header_len, "tensors": tensors}
    header_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def fetch_tensor_leading_slice(
    tensor_name: str,
    leading_count: int | None = None,
    *,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    url: str = HF_SAFETENSORS_URL,
) -> torch.Tensor:
    """Fetch an entire tensor or its first `leading_count` outer slices (e.g. 1 of 128 experts).

    For `[128, 2304, 1152]` NVFP4 `w13_weight` (324 MB total), `leading_count=1` downloads
    only Expert 0 (2.53 MB) over HTTP Range and returns shape `[1, 2304, 1152]`.
    """
    meta = fetch_safetensors_header(cache_dir=cache_dir, url=url)
    base_offset = 8 + int(meta["header_len"])
    tensors = meta["tensors"]
    if tensor_name not in tensors:
        raise KeyError(f"Tensor {tensor_name!r} not found in Inkling-2-Mini safetensors")

    info = tensors[tensor_name]
    dtype_str: str = info["dtype"]
    full_shape: list[int] = list(info["shape"])
    start_rel, end_rel = info["data_offsets"]
    torch_dtype, elem_bytes = _DTYPE_VIEW[dtype_str]

    if leading_count is not None and len(full_shape) > 0:
        if not (1 <= leading_count <= full_shape[0]):
            raise ValueError(
                f"leading_count={leading_count} out of bounds for shape {full_shape}"
            )
        inner_elems = 1
        for dim in full_shape[1:]:
            inner_elems *= dim
        nbytes = leading_count * inner_elems * elem_bytes
        out_shape = [leading_count, *full_shape[1:]]
    else:
        nbytes = end_rel - start_rel
        out_shape = full_shape

    slices_dir = cache_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{tensor_name}:{out_shape}:{dtype_str}".encode()).hexdigest()[:16]
    safe_stub = tensor_name.replace(".", "_")
    cache_file = slices_dir / f"{safe_stub}__{key}.bin"

    if cache_file.exists() and cache_file.stat().st_size == nbytes:
        raw = cache_file.read_bytes()
    else:
        abs_start = base_offset + start_rel
        raw = _http_range_get(url, abs_start, abs_start + nbytes - 1)
        if len(raw) != nbytes:
            raise RuntimeError(f"Expected {nbytes} bytes for {tensor_name}, got {len(raw)}")
        cache_file.write_bytes(raw)

    # View raw bytes as uint8 first (supports float8_e4m3fn cleanly across torch versions)
    u8 = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    return u8.view(torch_dtype).reshape(out_shape)


def fetch_real_nvfp4_expert_sample(
    layer_id: int = 0,
    num_experts: int = 2,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> dict[str, torch.Tensor]:
    """Fetch `num_experts` real NVFP4 experts from layers 0..19 (~8.5 MB for 2 experts)."""
    if not (0 <= layer_id <= 19):
        raise ValueError(f"NVFP4 layers are 0..19, got {layer_id}")
    prefix = f"model.llm.layers.{layer_id}.mlp.experts"
    return {
        "w13_weight": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight", num_experts, cache_dir=cache_dir
        ),
        "w13_weight_scale": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight_scale", num_experts, cache_dir=cache_dir
        ),
        "w13_weight_amax": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight_amax", num_experts, cache_dir=cache_dir
        ),
        "w13_denom": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight_outer_scale_denominator", None, cache_dir=cache_dir
        ),
        "w2_weight": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight", num_experts, cache_dir=cache_dir
        ),
        "w2_weight_scale": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight_scale", num_experts, cache_dir=cache_dir
        ),
        "w2_weight_amax": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight_amax", num_experts, cache_dir=cache_dir
        ),
        "w2_denom": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight_outer_scale_denominator", None, cache_dir=cache_dir
        ),
    }


def fetch_real_mxfp8_expert_sample(
    layer_id: int = 20,
    num_experts: int = 1,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> dict[str, torch.Tensor]:
    """Fetch `num_experts` real MXFP8 experts from layers 20..23 (~8.1 MB for 1 expert)."""
    if not (20 <= layer_id <= 23):
        raise ValueError(f"MXFP8 layers are 20..23, got {layer_id}")
    prefix = f"model.llm.layers.{layer_id}.mlp.experts"
    return {
        "w13_weight": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight", num_experts, cache_dir=cache_dir
        ),
        "w13_weight_scale": fetch_tensor_leading_slice(
            f"{prefix}.w13_weight_scale", num_experts, cache_dir=cache_dir
        ),
        "w2_weight": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight", num_experts, cache_dir=cache_dir
        ),
        "w2_weight_scale": fetch_tensor_leading_slice(
            f"{prefix}.w2_weight_scale", num_experts, cache_dir=cache_dir
        ),
    }


def summarize_checkpoint_precision_bytes(cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict[str, int]:
    """Sum exact byte footprint by dtype from the official safetensors header."""
    meta = fetch_safetensors_header(cache_dir=cache_dir)
    totals: dict[str, int] = {}
    for name, info in meta["tensors"].items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]
        s, e = info["data_offsets"]
        totals[dt] = totals.get(dt, 0) + (e - s)
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))
