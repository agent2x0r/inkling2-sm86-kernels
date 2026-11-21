#!/usr/bin/env python3
"""Publication-grade functional & performance benchmark suite for Inkling-2-Mini on SM86 vLLM.

Covers:
  1. Multimodal vision verification (patched `nvidia/vision.py` FA2 path on SM86)
  2. Structured tool-calling verification (`--tool-call-parser inkling`)
  3. Greedy (`temperature=0`, `seed=42`) 12-prompt parity capture (Base vs DSpark)
  4. Multi-trial context-length scaling (1K, 4K, 16K, 32K, 60K) with 0% prefix-cache hits
  5. Output-category breakdown (Code, Math, Technical Prose, JSON)
  6. Multi-user concurrency sweep (c = 1, 2, 4, 8) at 4K context
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import pathlib
import re
import statistics
import struct
import sysconfig
import time
import urllib.parse
import urllib.request
import zlib
from typing import Any

MODEL_ID = "thinkingmachines/Inkling-2-Mini"


# ---------------------------------------------------------------------------
# Pure-Python PNG generator (zero external dependencies)
# ---------------------------------------------------------------------------
def _make_png(width: int, height: int, rgb_fn) -> bytes:
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type None
        for x in range(width):
            r, g, b = rgb_fn(x, y)
            raw.extend((r, g, b))

    def chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
        + chunk(b"IEND", b"")
    )


def make_shapes_png() -> bytes:
    """Crisp 480x240 image: 3 solid red squares on top row, 2 solid blue circles on bottom."""
    squares = [(60, 30, 120, 90), (210, 30, 270, 90), (360, 30, 420, 90)]
    circles = [(150, 175, 32), (330, 175, 32)]

    def px(x: int, y: int) -> tuple[int, int, int]:
        for x0, y0, x1, y1 in squares:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return (220, 30, 30)
        for cx, cy, r in circles:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                return (30, 70, 225)
        return (255, 255, 255)

    return _make_png(480, 240, px)


def make_highres_checker_png() -> bytes:
    """High-res 1024x1024 4x4 colored grid to stress-test vision FA2 patch attention."""
    def px(x: int, y: int) -> tuple[int, int, int]:
        gx, gy = x // 256, y // 256
        if (gx + gy) % 2 == 0:
            return (20, 160, 60)  # green
        return (240, 200, 20)  # yellow

    return _make_png(1024, 1024, px)


# ---------------------------------------------------------------------------
# vLLM HTTP + Prometheus metrics helpers
# ---------------------------------------------------------------------------
def get_metrics(base_url: str) -> dict[str, float]:
    txt = urllib.request.urlopen(f"{base_url}/metrics", timeout=15).read().decode()
    out: dict[str, float] = {}
    for k in (
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
    ):
        m = re.search(rf"^{re.escape(k)}(?:\{{[^}}]*\}})?\s+([0-9eE.+-]+)", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


def chat_complete(
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.6,
    seed: int | None = None,
    stream: bool = True,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if seed is not None:
        payload["seed"] = seed
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if reasoning_effort is not None:
        payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode())
        t1 = time.perf_counter()
        msg = obj["choices"][0]["message"]
        usage = obj.get("usage") or {}
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "ttft_s": t1 - t0,
            "total_s": t1 - t0,
            "decode_tps": 0.0,
            "prefill_tps": 0.0,
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": obj["choices"][0].get("finish_reason"),
        }

    t_first: float | None = None
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            delta = choices[0].get("delta") or {}
            r = delta.get("reasoning") or delta.get("reasoning_content") or ""
            c = delta.get("content") or ""
            if (r or c) and t_first is None:
                t_first = time.perf_counter()
            if r:
                reasoning_parts.append(r)
            if c:
                content_parts.append(c)
    t1 = time.perf_counter()
    p_toks = int(usage.get("prompt_tokens", 0))
    c_toks = int(usage.get("completion_tokens", 0))
    ttft = (t_first - t0) if t_first is not None else (t1 - t0)
    gen_s = max(t1 - (t_first if t_first is not None else t0), 1e-6)
    decode_tps = (c_toks - 1) / gen_s if c_toks > 1 else c_toks / gen_s
    prefill_tps = p_toks / max(ttft, 1e-6)
    return {
        "prompt_tokens": p_toks,
        "completion_tokens": c_toks,
        "ttft_s": ttft,
        "total_s": t1 - t0,
        "decode_tps": decode_tps,
        "prefill_tps": prefill_tps,
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "finish_reason": finish_reason,
    }


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": round(statistics.mean(vals), 3),
        "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# 1. Multimodal Vision verification
# ---------------------------------------------------------------------------
def run_vision_tests(base_url: str) -> list[dict[str, Any]]:
    print("\n=== [1/6] Multimodal Vision Tests (SM86 FA2 path in nvidia/vision.py) ===", flush=True)
    results = []

    # Test 1A: 3 red squares + 2 blue circles
    b64_shapes = base64.b64encode(make_shapes_png()).decode()
    r1 = chat_complete(
        base_url,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_shapes}"}},
                {"type": "text", "text": "Count the exact number of red squares and blue circles in this image. Reply in the format: red_squares=X, blue_circles=Y."},
            ],
        }],
        max_tokens=1024,
        temperature=0.0,
        reasoning_effort="low",
    )
    ok1 = ("3" in r1["content"] and "2" in r1["content"] and "red_squares=3" in r1["content"].lower().replace(" ", ""))
    results.append({
        "name": "synthetic_shapes_480x240",
        "passed": ok1,
        "prompt_tokens": r1["prompt_tokens"],
        "ttft_s": round(r1["ttft_s"], 3),
        "decode_tps": round(r1["decode_tps"], 1),
        "content": r1["content"].strip()[:240],
    })
    print(f"  1A shapes (480x240): passed={ok1} | prompt_toks={r1['prompt_tokens']} | ttft={r1['ttft_s']:.2f}s | {r1['content'].strip()[:100]}", flush=True)

    # Test 1B: High-res 1024x1024 4x4 green/yellow checkerboard
    b64_grid = base64.b64encode(make_highres_checker_png()).decode()
    r2 = chat_complete(
        base_url,
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_grid}"}},
                {"type": "text", "text": "What two colors form this checkerboard grid, and how many total squares (rows x columns) are there? Be concise."},
            ],
        }],
        max_tokens=1024,
        temperature=0.0,
        reasoning_effort="low",
    )
    txt2 = r2["content"].lower()
    ok2 = ("green" in txt2 and "yellow" in txt2 and ("16" in txt2 or "4x4" in txt2 or "4 x 4" in txt2))
    results.append({
        "name": "highres_checkerboard_1024x1024",
        "passed": ok2,
        "prompt_tokens": r2["prompt_tokens"],
        "ttft_s": round(r2["ttft_s"], 3),
        "decode_tps": round(r2["decode_tps"], 1),
        "content": r2["content"].strip()[:240],
    })
    print(f"  1B highres (1024x1024): passed={ok2} | prompt_toks={r2['prompt_tokens']} | ttft={r2['ttft_s']:.2f}s | {r2['content'].strip()[:100]}", flush=True)

    # Test 1C: Real MMMU-Pro image fetched from HuggingFace
    try:
        q = urllib.parse.urlencode({"dataset": "MMMU/MMMU_Pro", "config": "standard (10 options)", "split": "test", "offset": 0, "length": 1})
        d = json.loads(urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=15).read())
        row = d["rows"][0]["row"]
        img_bytes = urllib.request.urlopen(row["image_1"]["src"], timeout=15).read()
        b64_mmmu = base64.b64encode(img_bytes).decode()
        opts = row["options"]
        if isinstance(opts, str):
            import ast
            opts = ast.literal_eval(opts)
        letters = "ABCDEFGHIJ"
        opt_lines = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(opts))
        r3 = chat_complete(
            base_url,
            [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_mmmu}"}},
                    {"type": "text", "text": f"{row['question']}\n\nOptions:\n{opt_lines}\n\nRespond with the final answer letter (A-J) at the end as 'Answer: <LETTER>'."},
                ],
            }],
            max_tokens=2048,
            temperature=0.0,
            reasoning_effort="medium",
        )
        m = re.search(r"Answer:\s*([A-J])\b", r3["content"], re.I)
        pred = m.group(1).upper() if m else ""
        gold = str(row["answer"]).strip().upper()
        ok3 = pred == gold
        results.append({
            "name": "mmmu_pro_real_image_smoke",
            "passed": ok3,
            "pred": pred,
            "gold": gold,
            "prompt_tokens": r3["prompt_tokens"],
            "ttft_s": round(r3["ttft_s"], 3),
            "decode_tps": round(r3["decode_tps"], 1),
            "content": r3["content"].strip()[:240],
        })
        print(f"  1C MMMU-Pro real image: passed={ok3} (pred={pred}, gold={gold}) | prompt_toks={r3['prompt_tokens']} | ttft={r3['ttft_s']:.2f}s", flush=True)
    except Exception as e:
        results.append({"name": "mmmu_pro_real_image_smoke", "passed": False, "error": str(e)})
        print(f"  1C MMMU-Pro real image ERR: {e}", flush=True)

    return results


# ---------------------------------------------------------------------------
# 2. Structured Tool-Calling verification
# ---------------------------------------------------------------------------
def run_tool_calling_tests(base_url: str) -> list[dict[str, Any]]:
    print("\n=== [2/6] Structured Tool-Calling Tests (--tool-call-parser inkling) ===", flush=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_gpu_telemetry",
                "description": "Fetch live VRAM usage, temperature, and power draw for a GPU index on a host.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hostname": {"type": "string", "description": "Host name or IP address"},
                        "gpu_index": {"type": "integer", "description": "0-based GPU index"},
                    },
                    "required": ["hostname", "gpu_index"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_bandwidth_gbs",
                "description": "Compute memory bandwidth in GB/s given bytes transferred and elapsed microseconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bytes_moved": {"type": "integer"},
                        "elapsed_us": {"type": "number"},
                    },
                    "required": ["bytes_moved", "elapsed_us"],
                },
            },
        },
    ]

    cases = [
        (
            "single_tool_gpu_telemetry",
            "Check the live GPU telemetry for GPU 1 on host inference-01.",
            "get_gpu_telemetry",
            {"hostname": "inference-01", "gpu_index": 1},
        ),
        (
            "single_tool_bandwidth_calc",
            "Use the bandwidth tool to compute GB/s for moving 65690000 bytes in 84.83 microseconds.",
            "compute_bandwidth_gbs",
            {"bytes_moved": 65690000, "elapsed_us": 84.83},
        ),
    ]
    out = []
    for name, prompt, expected_fn, expected_args in cases:
        r = chat_complete(
            base_url,
            [{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.0,
            stream=False,
            tools=tools,
            reasoning_effort="low",
        )
        tcs = r.get("tool_calls") or []
        passed = False
        parsed_fn = None
        parsed_args = None
        if tcs:
            fn = tcs[0].get("function") or {}
            parsed_fn = fn.get("name")
            try:
                parsed_args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                parsed_args = fn.get("arguments")
            if parsed_fn == expected_fn and isinstance(parsed_args, dict):
                passed = all(
                    (
                        math.isclose(float(parsed_args.get(k, -1)), float(v), rel_tol=1e-4)
                        if isinstance(v, (int, float))
                        else parsed_args.get(k) == v
                    )
                    for k, v in expected_args.items()
                )
        out.append({
            "name": name,
            "passed": passed,
            "fn": parsed_fn,
            "args": parsed_args,
            "finish_reason": r.get("finish_reason"),
        })
        print(f"  {name}: passed={passed} | fn={parsed_fn} | args={parsed_args}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 3. Deterministic Greedy Parity Suite (temp=0, seed=42)
# ---------------------------------------------------------------------------
GREEDY_PROMPTS = [
    "Compute 347 * 892 step by step and state the final integer.",
    "Write a Python function `is_prime(n: int) -> bool` using trial division up to sqrt(n).",
    "What is the time and space complexity of merge sort? Reply in 2 sentences.",
    "Return a valid JSON object with keys `arch`, `warp_size`, and `fp4_group_size` for Ampere SM86 NVFP4.",
    "If a train travels 180 km in 2 hours and 15 minutes, what is its average speed in km/h?",
    "Write a single SQL query returning the top 3 departments by average salary from table `employees(id, dept, salary)`.",
    "Give the exact hexadecimal IEEE-754 float32 representation of -6.5.",
    "Reverse the words in the sentence: 'Ampere dual GPUs love NVLink bandwidth'. Output only the reversed sentence.",
    "Factor the polynomial x^2 - 5x - 14 over the integers.",
    "Write a 4-line bash command that counts `.py` files recursively under the current directory.",
    "What is the derivative of f(x) = x^3 * ln(x)?",
    "List the first 10 Fibonacci numbers starting from 0, 1 separated by commas only.",
]


def run_greedy_suite(base_url: str) -> list[dict[str, Any]]:
    print("\n=== [3/6] Greedy (temp=0, seed=42) 12-Prompt Parity Capture ===", flush=True)
    items = []
    for i, p in enumerate(GREEDY_PROMPTS):
        r = chat_complete(
            base_url,
            [{"role": "user", "content": p}],
            max_tokens=512,
            temperature=0.0,
            seed=42,
            reasoning_effort="low",
        )
        items.append({
            "idx": i,
            "prompt": p,
            "completion_tokens": r["completion_tokens"],
            "reasoning": r["reasoning"],
            "content": r["content"],
            "decode_tps": round(r["decode_tps"], 1),
        })
    print(f"  Captured {len(items)} greedy responses (mean decode={statistics.mean(x['decode_tps'] for x in items):.1f} tok/s)", flush=True)
    return items


# ---------------------------------------------------------------------------
# 4. Multi-Trial Context-Length Scaling (1K, 4K, 16K, 32K, 60K)
# ---------------------------------------------------------------------------
def _load_stdlib_corpus() -> str:
    candidates = [
        pathlib.Path(sysconfig.get_path("stdlib")),
        pathlib.Path("/usr/lib/python3.12"),
        pathlib.Path("/usr/lib/python3.14"),
    ]
    stdlib = next((p for p in candidates if p.is_dir() and any(p.glob("*.py"))), None)
    if stdlib is None:
        raise FileNotFoundError(f"Could not locate Python stdlib .py files in {candidates}")
    py_files = sorted(
        [p for p in stdlib.glob("*.py") if p.stat().st_size > 8000],
        key=lambda p: p.name,
    )
    corpus = "\n".join(f"# === FILE: {p.name} ===\n" + p.read_text(errors="ignore") for p in py_files[:50])
    if len(corpus) < 200_000:
        raise RuntimeError(f"Stdlib corpus too short ({len(corpus)} chars) from {stdlib}")
    return corpus


def run_context_scaling(base_url: str, trials: int = 3) -> list[dict[str, Any]]:
    print(f"\n=== [4/6] Context-Length Scaling (1K..60K x {trials} trials, 0% prefix cache) ===", flush=True)
    corpus = _load_stdlib_corpus()
    # Char budgets tuned to hit ~1K, ~4K, ~16K, ~32K, ~60K real tokens
    targets = [
        ("1K", 3_100),
        ("4K", 12_800),
        ("16K", 52_000),
        ("32K", 104_000),
        ("60K", 192_000),
    ]
    summary = []
    for label, n_chars in targets:
        p_toks_list, ttft_list, prefill_list, decode_list, acc_list, phits_list = [], [], [], [], [], []
        needle_hits = 0
        for t in range(trials):
            nonce = int(time.time() * 100) % 100000 + t * 997 + n_chars % 97
            needle = f"CHK-{1000 + t * 313 + n_chars % 700}"
            text = corpus[:n_chars]
            mid = len(text) // 2
            # Use benign calibration label so alignment never refuses to report it
            doc = (
                f"# Benchmark session {nonce}\n"
                + text[:mid]
                + f"\n# BENCHMARK_CALIBRATION_TAG = \"{needle}\"\n"
                + text[mid:]
                + f"\n\n---\n1. State the exact BENCHMARK_CALIBRATION_TAG value from the code above.\n"
                f"2. Write a 6-sentence technical summary of Python standard library patterns visible in this excerpt."
            )
            m0 = get_metrics(base_url)
            r = chat_complete(
                base_url,
                [{"role": "user", "content": doc}],
                max_tokens=512,
                temperature=0.6,
                reasoning_effort="low",
            )
            m1 = get_metrics(base_url)
            d_drafts = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
            d_acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
            acc_len = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0
            p_hits = m1["vllm:prefix_cache_hits_total"] - m0["vllm:prefix_cache_hits_total"]
            hit = needle in (r["reasoning"] + "\n" + r["content"])
            needle_hits += int(hit)
            p_toks_list.append(r["prompt_tokens"])
            ttft_list.append(r["ttft_s"])
            prefill_list.append(r["prefill_tps"])
            decode_list.append(r["decode_tps"])
            acc_list.append(acc_len)
            phits_list.append(p_hits)

        row = {
            "bucket": label,
            "trials": trials,
            "prompt_tokens_mean": round(statistics.mean(p_toks_list)),
            "ttft_s": _mean_std(ttft_list),
            "prefill_tps": _mean_std(prefill_list),
            "decode_tps": _mean_std(decode_list),
            "spec_acc_len": _mean_std(acc_list),
            "prefix_cache_hits_total": int(sum(phits_list)),
            "needle_hits": f"{needle_hits}/{trials}",
        }
        summary.append(row)
        print(
            f"  {label:3s} (~{row['prompt_tokens_mean']:5d} tok) | TTFT={row['ttft_s']['mean']:5.2f}±{row['ttft_s']['std']:.2f}s | "
            f"prefill={row['prefill_tps']['mean']:6.1f}±{row['prefill_tps']['std']:5.1f} tok/s | "
            f"decode={row['decode_tps']['mean']:5.1f}±{row['decode_tps']['std']:4.1f} tok/s | "
            f"acc={row['spec_acc_len']['mean']:.2f} | prefix_hits={row['prefix_cache_hits_total']} | needle={row['needle_hits']}",
            flush=True,
        )
    return summary


# ---------------------------------------------------------------------------
# 5. Output-Category Breakdown (Code, Math, Technical Prose, JSON)
# ---------------------------------------------------------------------------
def run_category_breakdown(base_url: str, trials: int = 3) -> list[dict[str, Any]]:
    print(f"\n=== [5/6] Output Category Speed Breakdown (x {trials} trials, 512 out tok) ===", flush=True)
    categories = [
        ("Code (Python/CUDA)", "Implement a complete LRU cache with TTL expiration in Python including type hints, unit tests, and docstring comments."),
        ("Math Reasoning", "Prove that there are infinitely many primes congruent to 3 mod 4, then solve 7x ≡ 3 (mod 29) and verify the result."),
        ("Technical Prose", "Write an in-depth architecture note explaining the trade-offs between tensor parallelism, expert parallelism, and pipeline parallelism for sparse MoE models on NVLink dual-GPU nodes."),
        ("Structured JSON", "Generate a valid JSON array of 12 detailed synthetic GPU benchmark entries with fields: gpu_name, arch, vram_gb, bandwidth_gbs, fp16_tflops, tdp_w, release_year, notes."),
    ]
    rows = []
    for cat, base_prompt in categories:
        dec_list, acc_list, ttft_list = [], [], []
        for t in range(trials):
            nonce = int(time.time() * 100) % 100000 + t * 503
            m0 = get_metrics(base_url)
            r = chat_complete(
                base_url,
                [{"role": "user", "content": f"[trial={nonce}] {base_prompt}"}],
                max_tokens=512,
                temperature=0.6,
                reasoning_effort="low",
            )
            m1 = get_metrics(base_url)
            d_drafts = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
            d_acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
            acc_len = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0
            dec_list.append(r["decode_tps"])
            acc_list.append(acc_len)
            ttft_list.append(r["ttft_s"])
        row = {
            "category": cat,
            "decode_tps": _mean_std(dec_list),
            "spec_acc_len": _mean_std(acc_list),
            "ttft_s": _mean_std(ttft_list),
        }
        rows.append(row)
        print(
            f"  {cat:20s} | decode={row['decode_tps']['mean']:5.1f}±{row['decode_tps']['std']:4.1f} tok/s | "
            f"acc={row['spec_acc_len']['mean']:.2f}±{row['spec_acc_len']['std']:.2f} | ttft={row['ttft_s']['mean']*1000:.0f}ms",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
# 6. Multi-User Concurrency Sweep (c = 1, 2, 4, 8) at 4K context
# ---------------------------------------------------------------------------
def run_concurrency_sweep(base_url: str) -> list[dict[str, Any]]:
    print("\n=== [6/6] Multi-User Concurrency Sweep (c = 1, 2, 4, 8 at ~4K prompt, 512 out tok) ===", flush=True)
    corpus = _load_stdlib_corpus()
    rows = []
    for c in (1, 2, 4, 8):
        m0 = get_metrics(base_url)

        def worker(i: int) -> dict[str, Any]:
            nonce = int(time.time() * 100) % 100000 + c * 1000 + i * 73
            # Shift corpus offset per user so every user has a unique prompt (0% prefix sharing)
            off = (i * 1700) % 25000
            doc = (
                f"# User {i} session {nonce}\n"
                + corpus[off : off + 12_800]
                + "\n\n---\nSummarize the key classes and error-handling design patterns in this module in 6 technical sentences."
            )
            return chat_complete(
                base_url,
                [{"role": "user", "content": doc}],
                max_tokens=512,
                temperature=0.6,
                reasoning_effort="low",
            )

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(worker, i) for i in range(c)]
            res = [f.result() for f in futs]
        wall_s = time.perf_counter() - t0
        m1 = get_metrics(base_url)

        total_prompt = sum(r["prompt_tokens"] for r in res)
        total_comp = sum(r["completion_tokens"] for r in res)
        per_user_dec = [r["decode_tps"] for r in res]
        ttfts = sorted(r["ttft_s"] for r in res)
        d_drafts = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
        d_acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
        acc_len = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0
        p_hits = int(m1["vllm:prefix_cache_hits_total"] - m0["vllm:prefix_cache_hits_total"])

        # True aggregate decode throughput = total completion tokens / (wall - min_ttft)
        agg_decode_tps = total_comp / max(wall_s - ttfts[0], 1e-6)
        agg_total_tps = total_comp / max(wall_s, 1e-6)
        row = {
            "concurrency": c,
            "prompt_tokens_per_req": round(total_prompt / c),
            "comp_tokens_total": total_comp,
            "wall_s": round(wall_s, 2),
            "ttft_p50_s": round(statistics.median(ttfts), 2),
            "ttft_max_s": round(ttfts[-1], 2),
            "per_user_decode_tps": _mean_std(per_user_dec),
            "agg_decode_tps": round(agg_decode_tps, 1),
            "agg_e2e_gen_tps": round(agg_total_tps, 1),
            "spec_acc_len": round(acc_len, 2),
            "prefix_cache_hits": p_hits,
        }
        rows.append(row)
        print(
            f"  c={c} | wall={row['wall_s']:5.2f}s | TTFT p50/max={row['ttft_p50_s']:.2f}/{row['ttft_max_s']:.2f}s | "
            f"per-user={row['per_user_decode_tps']['mean']:5.1f} tok/s | "
            f"AGG decode={row['agg_decode_tps']:6.1f} tok/s | acc={row['spec_acc_len']:.2f} | hits={p_hits}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dspark", "base"), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-tools", action="store_true")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "mode": args.mode,
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not args.skip_vision:
        out["vision"] = run_vision_tests(args.base_url)
    if not args.skip_tools:
        out["tool_calling"] = run_tool_calling_tests(args.base_url)
    out["greedy"] = run_greedy_suite(args.base_url)
    out["context_scaling"] = run_context_scaling(args.base_url, trials=args.trials)
    out["categories"] = run_category_breakdown(args.base_url, trials=args.trials)
    out["concurrency"] = run_concurrency_sweep(args.base_url)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {args.out}", flush=True)


if __name__ == "__main__":
    main()
