#!/usr/bin/env python3
"""Accuracy evaluation harness for Inkling-2-Mini on SM86 vLLM.

Evaluates three public benchmarks directly against the live OpenAI API:
  1. HumanEval (all 164/164 problems, real isolated Python unit-test execution -> pass@1)
  2. GSM8K (test split subset, strict normalized numeric extraction -> exact match)
  3. MMMU-Pro Standard 10 (real multimodal image questions -> comparable to published 74.9%)
"""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
CACHE_DIR = pathlib.Path("/tmp/inkling2_eval_cache")


# ---------------------------------------------------------------------------
# HuggingFace datasets-server loader (zero external deps, disk-cached)
# ---------------------------------------------------------------------------
def fetch_hf_rows(dataset: str, config: str, split: str, count: int) -> list[dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{dataset}_{config}_{split}_{count}")
    cache_path = CACHE_DIR / f"{safe}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < count:
        length = min(100, count - len(rows))
        q = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
        )
        url = f"https://datasets-server.huggingface.co/rows?{q}"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < length:
            break

    cache_path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def call_model(
    base_url: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    reasoning_effort: str = "low",
    timeout: int = 300,
) -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode())
    msg = obj["choices"][0]["message"]
    return {
        "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
        "finish_reason": obj["choices"][0].get("finish_reason"),
        "usage": obj.get("usage") or {},
    }


# ---------------------------------------------------------------------------
# 1. HumanEval (Full 164/164)
# ---------------------------------------------------------------------------
def _extract_python_code(text: str, entry_point: str, prompt: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    if blocks:
        # Prefer the block defining entry_point if multiple blocks exist
        code = blocks[-1]
        for blk in reversed(blocks):
            if f"def {entry_point}" in blk:
                code = blk
                break
    else:
        code = text.strip()

    # If model returned full function (with `def entry_point`), preserve any helper
    # functions/imports from prompt above the target signature (e.g., HumanEval/38, /50).
    if f"def {entry_point}" in code:
        idx = prompt.find(f"def {entry_point}")
        prefix = prompt[:idx] if idx > 0 else ""
        return prefix + "\n" + code
    # Otherwise treat as indented function body appended to prompt
    if not code.startswith((" ", "\t")):
        code = "\n".join("    " + line if line.strip() else "" for line in code.splitlines())
    return prompt + "\n" + code


def _run_humaneval_test(program: str, test_code: str, entry_point: str) -> tuple[bool, str]:
    full = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(full)
        fname = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if proc.returncode == 0:
            return True, "ok"
        err = (proc.stderr or proc.stdout).strip().splitlines()
        return False, (err[-1] if err else f"rc={proc.returncode}")[:180]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        pathlib.Path(fname).unlink(missing_ok=True)


def eval_humaneval(base_url: str, count: int, workers: int) -> dict[str, Any]:
    rows = fetch_hf_rows("openai/openai_humaneval", "openai_humaneval", "test", count)
    print(f"\n=== [Accuracy 1/3] HumanEval ({len(rows)} problems, workers={workers}) ===", flush=True)
    t0 = time.perf_counter()

    def solve_one(row: dict[str, Any]) -> dict[str, Any]:
        prompt = row["prompt"]
        entry = row["entry_point"]
        user_msg = (
            "Complete the following Python function accurately. "
            "Output ONLY the complete Python function (including any needed imports) inside a single ```python``` code block.\n\n"
            f"```python\n{prompt}```"
        )
        r = call_model(
            base_url,
            [{"role": "user", "content": user_msg}],
            max_tokens=4096,
            temperature=0.2,
            reasoning_effort="low",
        )
        prog = _extract_python_code(r["content"], entry, prompt)
        passed, status = _run_humaneval_test(prog, row["test"], entry)
        return {
            "task_id": row["task_id"],
            "passed": passed,
            "status": status,
            "finish_reason": r["finish_reason"],
            "comp_tokens": r["usage"].get("completion_tokens", 0),
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, res in enumerate(ex.map(solve_one, rows), 1):
            results.append(res)
            if idx % 25 == 0 or idx == len(rows):
                n_pass = sum(x["passed"] for x in results)
                print(
                    f"  HumanEval progress: {idx}/{len(rows)} | pass@1={n_pass}/{idx} ({100.0*n_pass/idx:.1f}%) | elapsed={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )

    n_pass = sum(x["passed"] for x in results)
    elapsed = time.perf_counter() - t0
    tot_toks = sum(x["comp_tokens"] for x in results)
    failed = [x for x in results if not x["passed"]]
    return {
        "benchmark": "HumanEval",
        "total": len(results),
        "passed": n_pass,
        "pass_at_1_pct": round(100.0 * n_pass / max(len(results), 1), 2),
        "elapsed_s": round(elapsed, 1),
        "agg_comp_tps": round(tot_toks / max(elapsed, 1e-6), 1),
        "failures": failed[:20],
    }


# ---------------------------------------------------------------------------
# 2. GSM8K
# ---------------------------------------------------------------------------
def _norm_num(s: str) -> Decimal | None:
    s = s.strip().replace("\\$", "").replace("\\%", "").replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    # Strip trailing units (e.g. "160 minutes", "10 mph", "5 hours")
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?)", s)
    if m:
        s = m.group(1).strip()
    # Handle fractions like a/b
    if "/" in s and s.count("/") == 1:
        a, b = s.split("/")
        try:
            return Decimal(a.strip()) / Decimal(b.strip())
        except Exception:
            return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _extract_gsm8k_pred(text: str) -> str | None:
    # 1. \boxed{...} (last match)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    # 2. #### <num>
    hashes = re.findall(r"####\s*([-+]?[0-9,]+(?:\.[0-9]+)?)", text)
    if hashes:
        return hashes[-1].strip()
    # 3. Last number in final answer
    nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    return nums[-1] if nums else None


def eval_gsm8k(base_url: str, count: int, workers: int) -> dict[str, Any]:
    rows = fetch_hf_rows("openai/gsm8k", "main", "test", count)
    print(f"\n=== [Accuracy 2/3] GSM8K ({len(rows)} test problems, workers={workers}) ===", flush=True)
    t0 = time.perf_counter()

    def solve_one(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, row = idx_row
        gold_m = re.search(r"####\s*(.+)$", row["answer"], re.M)
        gold_str = gold_m.group(1).strip() if gold_m else ""
        gold_val = _norm_num(gold_str)

        r = call_model(
            base_url,
            [{
                "role": "user",
                "content": (
                    f"{row['question']}\n\n"
                    "Solve this math problem and put ONLY the final numeric answer inside \\boxed{}."
                ),
            }],
            max_tokens=4096,
            temperature=0.2,
            reasoning_effort="low",
        )
        pred_str = _extract_gsm8k_pred(r["content"])
        pred_val = _norm_num(pred_str) if pred_str else None
        ok = (gold_val is not None and pred_val is not None and abs(gold_val - pred_val) <= Decimal("1e-4"))
        return {
            "idx": i,
            "passed": ok,
            "gold": gold_str,
            "pred": pred_str,
            "finish_reason": r["finish_reason"],
            "comp_tokens": r["usage"].get("completion_tokens", 0),
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, res in enumerate(ex.map(solve_one, enumerate(rows)), 1):
            results.append(res)
            if idx % 25 == 0 or idx == len(rows):
                n_pass = sum(x["passed"] for x in results)
                print(
                    f"  GSM8K progress: {idx}/{len(rows)} | acc={n_pass}/{idx} ({100.0*n_pass/idx:.1f}%) | elapsed={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )

    n_pass = sum(x["passed"] for x in results)
    elapsed = time.perf_counter() - t0
    tot_toks = sum(x["comp_tokens"] for x in results)
    return {
        "benchmark": "GSM8K",
        "total": len(results),
        "passed": n_pass,
        "accuracy_pct": round(100.0 * n_pass / max(len(results), 1), 2),
        "elapsed_s": round(elapsed, 1),
        "agg_comp_tps": round(tot_toks / max(elapsed, 1e-6), 1),
        "failures": [x for x in results if not x["passed"]][:20],
    }


# ---------------------------------------------------------------------------
# 3. MMMU-Pro Standard 10 (Real multimodal vision benchmark)
# ---------------------------------------------------------------------------
def _extract_mc_letter(text: str, valid_letters: str) -> str | None:
    patterns = [
        r"\\boxed\{\s*([A-J])\s*\}",
        r"(?:final\s+answer|answer)\s*(?:is|:)\s*\**\(?([A-J])\)?\**\b",
        r"\*\*([A-J])\*\*",
        r"^\s*\(?([A-J])\)?[\.\s]*$",
    ]
    for pat in patterns:
        ms = re.findall(pat, text, re.I | re.M)
        if ms:
            cand = ms[-1].upper()
            if cand in valid_letters:
                return cand
    # Last standalone capital letter in valid_letters
    ms = re.findall(r"\b([A-J])\b", text)
    for cand in reversed(ms):
        if cand in valid_letters:
            return cand
    return None


def eval_mmmu_pro(base_url: str, count: int, workers: int, local_json: str | None = None) -> dict[str, Any]:
    if local_json:
        rows = json.loads(pathlib.Path(local_json).read_text(encoding="utf-8"))[:count]
    else:
        rows = fetch_hf_rows("MMMU/MMMU_Pro", "standard (10 options)", "test", count)
    print(f"\n=== [Accuracy 3/3] MMMU-Pro Standard 10 ({len(rows)} multimodal items, workers={workers}) ===", flush=True)
    t0 = time.perf_counter()
    letters = "ABCDEFGHIJ"

    def solve_one(idx_row: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        i, row = idx_row
        opts = row["options"]
        if isinstance(opts, str):
            opts = ast.literal_eval(opts)
        valid = letters[: len(opts)]
        opt_lines = "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(opts))

        content_parts: list[dict[str, Any]] = []
        n_imgs = 0
        if "images" in row and isinstance(row["images"], list):
            for data_uri in row["images"]:
                content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                n_imgs += 1
        else:
            for k in range(1, 8):
                img = row.get(f"image_{k}")
                if not img or not isinstance(img, dict) or not img.get("src"):
                    continue
                try:
                    img_req = urllib.request.Request(img["src"], headers={"User-Agent": "Mozilla/5.0"})
                    raw = urllib.request.urlopen(img_req, timeout=20).read()
                    mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
                    b64 = base64.b64encode(raw).decode()
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
                    n_imgs += 1
                except Exception as e:
                    return {"idx": i, "id": row.get("id"), "passed": False, "error": f"img_dl: {e}", "comp_tokens": 0}

        content_parts.append({
            "type": "text",
            "text": (
                f"{row['question']}\n\nOptions:\n{opt_lines}\n\n"
                f"Carefully inspect the image(s) and choose the single best option ({valid[0]}-{valid[-1]}). "
                "End your response with exactly: Answer: <LETTER>"
            ),
        })
        try:
            r = call_model(
                base_url,
                [{"role": "user", "content": content_parts}],
                max_tokens=4096,
                temperature=0.2,
                reasoning_effort="medium",
            )
        except Exception as e:
            return {"idx": i, "id": row.get("id"), "passed": False, "error": f"api: {e}", "comp_tokens": 0}

        gold = str(row["answer"]).strip().upper()
        pred = _extract_mc_letter(r["content"], valid)
        return {
            "idx": i,
            "id": row.get("id"),
            "subject": row.get("subject"),
            "n_imgs": n_imgs,
            "passed": pred == gold,
            "gold": gold,
            "pred": pred,
            "finish_reason": r["finish_reason"],
            "comp_tokens": r["usage"].get("completion_tokens", 0),
        }

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, res in enumerate(ex.map(solve_one, enumerate(rows)), 1):
            results.append(res)
            if idx % 20 == 0 or idx == len(rows):
                n_pass = sum(x["passed"] for x in results)
                print(
                    f"  MMMU-Pro progress: {idx}/{len(rows)} | acc={n_pass}/{idx} ({100.0*n_pass/idx:.1f}%) | elapsed={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )

    n_pass = sum(x["passed"] for x in results)
    elapsed = time.perf_counter() - t0
    tot_toks = sum(x["comp_tokens"] for x in results)
    return {
        "benchmark": "MMMU-Pro Standard 10",
        "published_ref_pct": 74.9,
        "total": len(results),
        "passed": n_pass,
        "accuracy_pct": round(100.0 * n_pass / max(len(results), 1), 2),
        "elapsed_s": round(elapsed, 1),
        "agg_comp_tps": round(tot_toks / max(elapsed, 1e-6), 1),
        "failures": [x for x in results if not x["passed"]][:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--humaneval", type=int, default=164)
    ap.add_argument("--gsm8k", type=int, default=200)
    ap.add_argument("--mmmu", type=int, default=100)
    ap.add_argument("--mmmu-json", default=None)
    args = ap.parse_args()

    out: dict[str, Any] = {
        "model": MODEL_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": {},
    }
    if args.humaneval > 0:
        out["benchmarks"]["humaneval"] = eval_humaneval(args.base_url, args.humaneval, args.workers)
    if args.gsm8k > 0:
        out["benchmarks"]["gsm8k"] = eval_gsm8k(args.base_url, args.gsm8k, args.workers)
    if args.mmmu > 0:
        out["benchmarks"]["mmmu_pro"] = eval_mmmu_pro(
            args.base_url, args.mmmu, min(args.workers, 6), local_json=args.mmmu_json
        )

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {args.out}", flush=True)


if __name__ == "__main__":
    main()
