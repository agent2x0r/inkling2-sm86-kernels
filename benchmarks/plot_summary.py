#!/usr/bin/env python3
"""Generate the dark-theme benchmark summary chart directly from checked-in JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    bench_dir = Path(__file__).resolve().parent
    base = json.loads((bench_dir / "results_base.json").read_text(encoding="utf-8"))
    dspark = json.loads((bench_dir / "results_dspark.json").read_text(encoding="utf-8"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 1.0,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=220)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")
        ax.grid(True, linestyle="--", alpha=0.22, color="#94a3b8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # --- Panel 1: Context-Length Scaling ---
    ctx_base = base["context_scaling"]
    ctx_dsp = dspark["context_scaling"]
    ctx_labels = [
        f"{r['bucket']}\n({r['prompt_tokens_mean']:,})"
        if r["prompt_tokens_mean"] < 1000
        else f"{r['bucket']}\n({r['prompt_tokens_mean']/1000:.1f}K)"
        for r in ctx_dsp
    ]
    x = np.arange(len(ctx_labels))
    base_mean = [r["decode_tps"]["mean"] for r in ctx_base]
    base_std = [r["decode_tps"]["std"] for r in ctx_base]
    dsp_mean = [r["decode_tps"]["mean"] for r in ctx_dsp]
    dsp_std = [r["decode_tps"]["std"] for r in ctx_dsp]
    prefill_k = [r["prefill_tps"]["mean"] / 1000.0 for r in ctx_dsp]

    ax1.errorbar(
        x,
        dsp_mean,
        yerr=dsp_std,
        marker="o",
        linewidth=2.6,
        markersize=7.5,
        capsize=4,
        color="#38bdf8",
        ecolor="#7dd3fc",
        label="vLLM + DSpark (7 draft tok)",
    )
    ax1.errorbar(
        x,
        base_mean,
        yerr=base_std,
        marker="s",
        linewidth=2.4,
        markersize=6.8,
        capsize=4,
        color="#f59e0b",
        ecolor="#fcd34d",
        label="vLLM Base (no spec decode)",
    )

    for i in range(len(x)):
        ax1.annotate(
            f"{dsp_mean[i]:.0f}",
            (x[i], dsp_mean[i] + dsp_std[i] + 6),
            ha="center",
            va="bottom",
            color="#7dd3fc",
            fontsize=9.5,
            fontweight="bold",
        )
        ax1.annotate(
            f"{base_mean[i]:.1f}",
            (x[i], base_mean[i] - 12),
            ha="center",
            va="top",
            color="#fcd34d",
            fontsize=9.2,
            fontweight="bold",
        )
        ax1.annotate(
            f"Prefill:\n{prefill_k[i]:.1f}k/s",
            (x[i], 30),
            ha="center",
            va="center",
            color="#94a3b8",
            fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.22", fc="#1e293b", ec="#334155", lw=0.7),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(ctx_labels)
    ax1.set_ylim(0, 275)
    ax1.set_ylabel("Single-User Decode Speed (tok/s)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Uncached Prompt Length (0% prefix hits, 3 trials ± 1σ)", fontsize=10)
    ax1.set_title("Context-Length Scaling (30/30 Needles Recalled)", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#f8fafc",
        fontsize=9.2,
    )

    # --- Panel 2: Output Category Decode Speed ---
    cat_order = [
        ("Structured JSON", "Structured\nJSON"),
        ("Code (Python/CUDA)", "Code\n(Py/CUDA)"),
        ("Math Reasoning", "Math\nReasoning"),
        ("Technical Prose", "Technical\nProse"),
    ]
    base_by_cat = {r["category"]: r for r in base["categories"]}
    dsp_by_cat = {r["category"]: r for r in dspark["categories"]}

    y = np.arange(len(cat_order))
    cat_base = [base_by_cat[k]["decode_tps"]["mean"] for k, _ in cat_order]
    cat_dsp = [dsp_by_cat[k]["decode_tps"]["mean"] for k, _ in cat_order]
    cat_dstd = [dsp_by_cat[k]["decode_tps"]["std"] for k, _ in cat_order]
    cat_acc = [dsp_by_cat[k]["spec_acc_len"]["mean"] for k, _ in cat_order]
    h = 0.34

    ax2.barh(
        y - h / 2,
        cat_dsp,
        height=h,
        xerr=cat_dstd,
        capsize=3.5,
        color="#38bdf8",
        ecolor="#bae6fd",
        label="vLLM + DSpark",
    )
    ax2.barh(y + h / 2, cat_base, height=h, color="#f59e0b", label="vLLM Base")

    for i in range(len(cat_order)):
        ax2.text(
            cat_dsp[i] + cat_dstd[i] + 4,
            y[i] - h / 2,
            f"{cat_dsp[i]:.1f} tok/s ({cat_dsp[i]/cat_base[i]:.2f}x, acc={cat_acc[i]:.2f})",
            va="center",
            ha="left",
            color="#e0f2fe",
            fontsize=8.8,
            fontweight="bold",
        )
        ax2.text(
            cat_base[i] + 4,
            y[i] + h / 2,
            f"{cat_base[i]:.1f} tok/s",
            va="center",
            ha="left",
            color="#fde68a",
            fontsize=8.8,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([lbl for _, lbl in cat_order])
    ax2.invert_yaxis()
    ax2.set_xlim(0, 380)
    ax2.set_xlabel("Decode Throughput (tok/s, 512 out tok, 3 trials ± 1σ)", fontsize=10)
    ax2.set_title("Decode Speed by Workload Entropy", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#f8fafc",
        fontsize=9.2,
    )

    fig.suptitle(
        "Inkling-2-Mini (26.4B MoE, NVFP4/MXFP8) on Dual RTX 3090 (SM86, 225W Cap, NVLink, Upstream vLLM V1)",
        color="#f8fafc",
        fontsize=13.2,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.016,
        "SM86 DSpark: c=1..8 agg 191→383 tok/s (4K) / 697 tok/s (coding)  •  HumanEval 97.6% (160/164)  •  "
        "GSM8K 98.0% (196/200)  •  Vision FA2 cos=0.999998",
        ha="center",
        color="#94a3b8",
        fontsize=8.6,
    )

    plt.tight_layout(rect=[0.01, 0.048, 0.99, 0.93])
    out_path = bench_dir / "inkling2_sm86_summary.png"
    plt.savefig(out_path, dpi=220, facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
