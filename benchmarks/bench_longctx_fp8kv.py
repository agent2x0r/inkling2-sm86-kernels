#!/usr/bin/env python3
"""Long-context needle recall & speed sweep (64K..320K+) for Inkling-2-Mini FP8 KV on SM86.

Reproduces `benchmarks/results_longctx_fp8kv.json` with verified `prefix_cache_hits = 0`
against a running vLLM OpenAI server.

Usage:
    python3 benchmarks/bench_longctx_fp8kv.py \
      --base-url http://127.0.0.1:8001 \
      --out benchmarks/results_longctx_fp8kv.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import sysconfig
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from suite_functional_and_speed import chat_complete, get_metrics


def load_large_corpus(min_chars: int = 1_500_000) -> str:
    """Assemble deterministic Python stdlib source text exceeding `min_chars`."""
    candidates = [
        pathlib.Path(sysconfig.get_path("stdlib")),
        pathlib.Path("/usr/lib/python3.12"),
        pathlib.Path("/usr/lib/python3.14"),
    ]
    stdlib = next((p for p in candidates if p.is_dir() and any(p.glob("*.py"))), None)
    if stdlib is None:
        raise FileNotFoundError(f"Could not locate Python stdlib .py files in {candidates}")

    py_files = sorted(
        [p for p in stdlib.rglob("*.py") if p.stat().st_size > 6000 and "__pycache__" not in str(p)],
        key=lambda p: str(p),
    )
    parts: list[str] = []
    total = 0
    for p in py_files:
        txt = f"\n# === FILE: {p.relative_to(stdlib)} ===\n" + p.read_text(errors="ignore")
        parts.append(txt)
        total += len(txt)
        if total >= min_chars:
            break
    corpus = "\n".join(parts)
    if len(corpus) < min_chars:
        base = corpus
        copy_i = 1
        while len(corpus) < min_chars:
            corpus += f"\n# === ARCHIVE COPY {copy_i} ===\n" + base
            copy_i += 1
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument(
        "--buckets",
        default="64K:205000,128K:410000,200K:640000,256K:820000,320K:1024000",
        help="Comma-separated label:char_budget entries",
    )
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", default="benchmarks/results_longctx_fp8kv.json")
    args = ap.parse_args()

    targets: list[tuple[str, int]] = []
    for item in args.buckets.split(","):
        lbl, chars = item.split(":")
        targets.append((lbl.strip(), int(chars)))

    corpus = load_large_corpus(max(c for _, c in targets) + 10_000)
    print(f"Loaded {len(corpus):,} chars of Python stdlib corpus.", flush=True)

    rows = []
    for idx, (label, n_chars) in enumerate(targets):
        nonce = int(time.time() * 100) % 100000 + idx * 997
        needle = f"FP8CHK-{4100 + idx * 317}"
        text = corpus[:n_chars]
        mid = len(text) // 2
        doc = (
            f"# Long-context FP8 KV session {nonce}\n"
            + text[:mid]
            + f'\n# BENCHMARK_CALIBRATION_TAG = "{needle}"\n'
            + text[mid:]
            + "\n\n---\n1. State the exact BENCHMARK_CALIBRATION_TAG value from the code above.\n"
            "2. Write a 6-sentence technical summary of Python standard library patterns visible in this excerpt."
        )
        m0 = get_metrics(args.base_url)
        r = chat_complete(
            args.base_url,
            [{"role": "user", "content": doc}],
            max_tokens=args.max_tokens,
            temperature=0.6,
            reasoning_effort="low",
            timeout=600,
        )
        m1 = get_metrics(args.base_url)
        d_drafts = m1["vllm:spec_decode_num_drafts_total"] - m0["vllm:spec_decode_num_drafts_total"]
        d_acc = m1["vllm:spec_decode_num_accepted_tokens_total"] - m0["vllm:spec_decode_num_accepted_tokens_total"]
        acc_len = (1.0 + d_acc / d_drafts) if d_drafts > 0 else 1.0
        p_hits = int(m1["vllm:prefix_cache_hits_total"] - m0["vllm:prefix_cache_hits_total"])
        hit = needle in (r["reasoning"] + "\n" + r["content"])
        row = {
            "bucket": label,
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "ttft_s": round(r["ttft_s"], 2),
            "prefill_tps": round(r["prefill_tps"], 1),
            "decode_tps": round(r["decode_tps"], 1),
            "spec_acc_len": round(acc_len, 2),
            "prefix_cache_hits": p_hits,
            "needle_hit": hit,
            "needle": needle,
        }
        rows.append(row)
        print(
            f"  {label:4s} ({row['prompt_tokens']:6d} tok) | TTFT={row['ttft_s']:6.2f}s | "
            f"prefill={row['prefill_tps']:6.1f} tok/s | decode={row['decode_tps']:5.1f} tok/s | "
            f"acc={row['spec_acc_len']:.2f} | hits={p_hits} | needle={'PASS' if hit else 'FAIL'}",
            flush=True,
        )

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"long_context_fp8": rows}, indent=2), encoding="utf-8")
    print(f"[SAVED] {out_path}", flush=True)


if __name__ == "__main__":
    main()
