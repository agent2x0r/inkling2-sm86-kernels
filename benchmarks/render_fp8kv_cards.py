#!/usr/bin/env python3
"""Render the dark-slate editorial Bento summary card + 3 X-ready tables from checked-in JSONs.

Usage:
    uv run --with matplotlib python3 benchmarks/render_fp8kv_cards.py

Outputs:
    benchmarks/inkling2_sm86_fp8kv_bento.png
    benchmarks/fp8kv_tables/01_fp8kv_capacity_accuracy.png
    benchmarks/fp8kv_tables/02_fp8kv_longctx_scaling.png
    benchmarks/fp8kv_tables/03_fp8kv_kernel_microbench.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Palette matching benchmarks/x_article_tables/ & plot_summary.py
BG = "#0b1220"
CARD_BG = "#111a2c"
CARD_ALT = "#152036"
BORDER = "#24344e"
CYAN = "#38bdf8"
CYAN_SOFT = "#7dd3fc"
EMERALD = "#34d399"
AMBER = "#f59e0b"
AMBER_SOFT = "#fcd34d"
TEXT = "#f8fafc"
TEXT_SUB = "#cbd5e1"
MUTED = "#8da0ba"
ROW_ODD = "#0e1626"
ROW_EVEN = "#131e33"
HEADER_BG = "#16243d"


def _add_card(ax, x, y, w, h, *, fc=CARD_BG, ec=BORDER, lw=1.1, radius=0.014):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    return box


def _add_pill(
    ax,
    x,
    y,
    text,
    *,
    fg=EMERALD,
    bg="#0d2b2a",
    ec="#1b5e50",
    fontsize=9.2,
    pad=0.32,
):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        color=fg,
        va="center",
        ha="left",
        bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.35", fc=bg, ec=ec, lw=0.9),
    )


def _fmt_ratio_str(raw: str) -> str:
    if "/" not in raw:
        return raw
    a, b = raw.split("/", 1)
    try:
        return f"{int(a):,}/{int(b):,}"
    except ValueError:
        return raw


def render_bento(
    dspark_fp8: dict,
    longctx: dict,
    acc_fp8: dict,
    kernel_json: dict,
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": BORDER,
            "axes.linewidth": 0.9,
        }
    )

    fig = plt.figure(figsize=(16.0, 9.6), dpi=220, facecolor=BG)
    canvas = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")

    # ------------------------------------------------------------------
    # Masthead (clean, bold, mobile-scannable)
    # ------------------------------------------------------------------
    canvas.text(
        0.028,
        0.955,
        "DUAL RTX 3090 (SM86, NVLINK)  ·  INKLING-2-MINI  ·  UPSTREAM vLLM V1",
        fontsize=10.4,
        fontweight="bold",
        color=CYAN,
        family="DejaVu Sans Mono",
        va="center",
    )
    canvas.text(
        0.028,
        0.913,
        "Software FP8 KV Cache on Ampere: 2× Context, Faster Decode",
        fontsize=23.5,
        fontweight="bold",
        color=TEXT,
        va="center",
    )
    canvas.text(
        0.028,
        0.875,
        "Unmodified official NVFP4/MXFP8 safetensors + 7-token DSpark  ·  Zero C++ compile  ·  Zero accuracy drop",
        fontsize=11.8,
        color=TEXT_SUB,
        va="center",
    )

    # ------------------------------------------------------------------
    # Row 1: 3 Big, Airy Hero Cards (y = 0.500 .. 0.840, h = 0.340)
    # ------------------------------------------------------------------
    y1, h1 = 0.500, 0.340
    gap = 0.016
    w_col = (1.0 - 2 * 0.028 - 2 * gap) / 3.0
    xA = 0.028
    xB = xA + w_col + gap
    xC = xB + w_col + gap

    # === CARD A: Context Capacity ===
    _add_card(canvas, xA, y1, w_col, h1)
    canvas.text(xA + 0.018, y1 + h1 - 0.030, "CONTEXT CAPACITY", fontsize=9.8, fontweight="bold", color=MUTED, family="DejaVu Sans Mono")
    canvas.text(xA + 0.018, y1 + h1 - 0.095, "482.8K", fontsize=39, fontweight="bold", color=TEXT, va="center")
    _add_pill(canvas, xA + 0.186, y1 + h1 - 0.090, "2.02× BF16", fg=EMERALD, bg="#0d2b2a", ec="#1d6b57", fontsize=11.2, pad=0.30)
    canvas.text(xA + 0.018, y1 + h1 - 0.145, "Unified KV pool  ·  416K max context", fontsize=10.8, color=TEXT_SUB)

    axA = fig.add_axes([xA + 0.018, y1 + 0.056, w_col - 0.036, 0.118], facecolor=CARD_BG)
    axA.set_xlim(0, 540)
    axA.set_ylim(-0.55, 1.60)
    axA.axis("off")
    bars_a = [
        (1, 482.81, CYAN, "SW FP8 KV", "482.8K tok · 0 pad layers"),
        (0, 239.06, AMBER, "BF16 KV", "239.1K tok · 18 pad layers"),
    ]
    for yi, val, col, lbl, note in bars_a:
        axA.barh(yi, 520, height=0.42, color="#1a2740", edgecolor="none")
        axA.barh(yi, val, height=0.42, color=col, edgecolor="none")
        axA.text(4, yi + 0.32, lbl, fontsize=10.0, fontweight="bold", color=TEXT, va="bottom")
        axA.text(520, yi + 0.32, note, fontsize=9.4, color=CYAN_SOFT if col == CYAN else AMBER_SOFT, va="bottom", ha="right")

    canvas.text(
        xA + 0.018,
        y1 + 0.024,
        "1:1 page match with BF16 draft removes all 18 pad layers",
        fontsize=9.4,
        color=MUTED,
        va="bottom",
    )

    # === CARD B: Attention Kernel Speedup ===
    _add_card(canvas, xB, y1, w_col, h1)
    regs = kernel_json["regimes"]
    dec_regs = [r for r in regs if r["q_len"] == 1]
    ver_regs = [r for r in regs if r["q_len"] == 8]
    pref_reg = [r for r in regs if r["q_len"] == 256][0]
    ver_lo = min(r["speedup"] for r in ver_regs)
    ver_hi = max(r["speedup"] for r in ver_regs)
    peak_dec = max(dec_regs, key=lambda r: r["speedup"])

    canvas.text(xB + 0.018, y1 + h1 - 0.030, "SM86 KERNEL SPEEDUP (L2-FLUSHED)", fontsize=9.8, fontweight="bold", color=MUTED, family="DejaVu Sans Mono")
    canvas.text(xB + 0.018, y1 + h1 - 0.095, f"{peak_dec['speedup']:.2f}×", fontsize=39, fontweight="bold", color=CYAN, va="center")
    _add_pill(canvas, xB + 0.146, y1 + h1 - 0.090, f"{peak_dec['fp8_read_gbs']:.0f} GB/s", fg=CYAN_SOFT, bg="#0d253a", ec="#1f5a85", fontsize=11.2, pad=0.30)
    canvas.text(xB + 0.018, y1 + h1 - 0.145, "Faster decode @ 128K  ·  0.00 diff vs dequant", fontsize=10.6, color=TEXT_SUB)

    axB = fig.add_axes([xB + 0.048, y1 + 0.072, w_col - 0.068, 0.114], facecolor=CARD_BG)
    xs = np.arange(len(dec_regs))
    bw_bar = 0.35
    bf_us = [r["bf16_us"] for r in dec_regs]
    fp_us = [r["fp8_us"] for r in dec_regs]
    axB.bar(xs - bw_bar / 2, bf_us, width=bw_bar, color=AMBER, label="BF16 (µs)")
    axB.bar(xs + bw_bar / 2, fp_us, width=bw_bar, color=CYAN, label="FP8 (µs)")
    for i, r in enumerate(dec_regs):
        axB.text(xs[i] - bw_bar / 2, bf_us[i] + 10, f"{bf_us[i]:.0f}", ha="center", va="bottom", fontsize=8.8, color=AMBER_SOFT, fontweight="bold")
        axB.text(xs[i] + bw_bar / 2, fp_us[i] + 10, f"{fp_us[i]:.0f}", ha="center", va="bottom", fontsize=8.8, color=CYAN_SOFT, fontweight="bold")
        axB.text(xs[i], max(bf_us[i], fp_us[i]) + 72, f"{r['speedup']:.2f}×", ha="center", va="bottom", fontsize=9.6, color=EMERALD, fontweight="bold")
    axB.set_xticks(xs)
    axB.set_xticklabels([f"{r['seqlen_k']//1024}K" for r in dec_regs], fontsize=9.5, fontweight="bold", color=TEXT_SUB)
    axB.set_ylim(0, max(bf_us) * 1.60)
    axB.set_yticks([0, 200, 400])
    axB.tick_params(axis="y", colors=MUTED, labelsize=8.8)
    axB.set_ylabel("µs", fontsize=9.2, color=MUTED)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.grid(axis="y", linestyle="--", alpha=0.18, color=MUTED)
    axB.legend(loc="upper left", frameon=False, fontsize=8.8, labelcolor=TEXT_SUB, ncol=2)

    canvas.text(
        xB + 0.018,
        y1 + 0.022,
        f"DSpark verify (q=8): {ver_lo:.2f}×–{ver_hi:.2f}×  ·  32K prefill: {pref_reg['speedup']:.2f}×",
        fontsize=9.4,
        color=MUTED,
        va="bottom",
    )

    # === CARD C: Quality & Functional Parity ===
    _add_card(canvas, xC, y1, w_col, h1)
    he = acc_fp8["benchmarks"]["humaneval"]
    gsm = acc_fp8["benchmarks"]["gsm8k"]
    c8_fp8 = [r for r in dspark_fp8["concurrency"] if r["concurrency"] == 8][0]["agg_decode_tps"]

    canvas.text(xC + 0.018, y1 + h1 - 0.030, "QUALITY & PARITY CHECKS", fontsize=9.8, fontweight="bold", color=MUTED, family="DejaVu Sans Mono")
    canvas.text(xC + 0.018, y1 + h1 - 0.095, f"{he['pass_at_1_pct']:.1f}%", fontsize=39, fontweight="bold", color=EMERALD, va="center")
    _add_pill(canvas, xC + 0.172, y1 + h1 - 0.090, "20/20 NIAH", fg=EMERALD, bg="#0d2b2a", ec="#1d6b57", fontsize=11.2, pad=0.30)
    canvas.text(xC + 0.018, y1 + h1 - 0.145, "HumanEval pass@1  ·  Zero accuracy loss", fontsize=10.8, color=TEXT_SUB)

    stats = [
        ("HumanEval", f"{he['passed']}/{he['total']} ({he['pass_at_1_pct']:.1f}%)", "BF16 97.6%"),
        ("GSM8K", f"{gsm['passed']}/{gsm['total']} ({gsm['accuracy_pct']:.1f}%)", "BF16 98.0%"),
        ("Needles", "20 / 20 (100%)", "1K → 337K"),
        ("Vision & Tools", "2/2  ·  2/2", f"c=8: {c8_fp8:.0f}/s"),
    ]
    row_y = y1 + 0.156
    for idx, (k, v, ref) in enumerate(stats):
        ry = row_y - idx * 0.037
        if idx % 2 == 0:
            _add_card(canvas, xC + 0.016, ry - 0.015, w_col - 0.032, 0.031, fc=CARD_ALT, ec="none", lw=0, radius=0.006)
        canvas.text(xC + 0.025, ry, k, fontsize=10.2, color=TEXT_SUB, va="center")
        canvas.text(xC + w_col - 0.098, ry, v, fontsize=10.4, fontweight="bold", color=TEXT, va="center", ha="right")
        canvas.text(xC + w_col - 0.020, ry, ref, fontsize=9.2, color=MUTED, va="center", ha="right")

    # ------------------------------------------------------------------
    # Row 2: Wide Left Chart (Long-Context 1K..337K) + Right "What Changed"
    # ------------------------------------------------------------------
    y2, h2 = 0.052, 0.424
    w_left = 0.618
    w_right = 1.0 - 2 * 0.028 - gap - w_left
    xL = 0.028
    xR = xL + w_left + gap

    # === CARD D: Long-Context Scaling (1K -> 336.6K) ===
    _add_card(canvas, xL, y2, w_left, h2)
    canvas.text(xL + 0.018, y2 + h2 - 0.030, "END-TO-END DECODE SCALING (DSPARK, 0% PREFIX HITS)", fontsize=9.8, fontweight="bold", color=MUTED, family="DejaVu Sans Mono")
    canvas.text(xL + 0.018, y2 + h2 - 0.068, "182 → 98.1 tok/s out to 336.6K tokens", fontsize=14.8, fontweight="bold", color=TEXT)

    short_rows = [r for r in dspark_fp8["context_scaling"] if r["bucket"] in ("1K", "4K", "16K", "32K")]
    long_rows = longctx["long_context_fp8"]
    pts = []
    for r in short_rows:
        pts.append((r["bucket"], r["prompt_tokens_mean"], r["decode_tps"]["mean"]))
    for r in long_rows:
        pts.append((r["bucket"], r["prompt_tokens"], r["decode_tps"]))

    axD = fig.add_axes([xL + 0.056, y2 + 0.098, w_left - 0.080, h2 - 0.205], facecolor=CARD_BG)
    x_idx = np.arange(len(pts))
    dec_vals = [p[2] for p in pts]
    tok_labels = [f"{p[0]}\n({p[1]/1000:.0f}K)" if p[1] >= 10000 else f"{p[0]}\n({p[1]/1000:.1f}K)" for p in pts]

    # Shade region beyond BF16 213K ceiling
    axD.axvspan(6.42, len(pts) - 0.45, color="#0d293e", alpha=0.60, zorder=0)
    axD.axvline(6.42, color=AMBER, linestyle="--", linewidth=1.6, zorder=2)
    axD.text(
        6.52,
        232,
        "Old BF16 ceiling (213K)\nFP8 unlocks → 416K",
        fontsize=9.6,
        fontweight="bold",
        color=AMBER_SOFT,
        va="top",
        ha="left",
    )

    bar_colors = [CYAN if i < 7 else EMERALD for i in range(len(pts))]
    axD.bar(x_idx, dec_vals, width=0.56, color=bar_colors, alpha=0.90, zorder=3)
    axD.plot(x_idx, dec_vals, color=TEXT, marker="o", markersize=5.2, linewidth=1.8, alpha=0.92, zorder=4)

    for i, (_, _, dec) in enumerate(pts):
        x_lbl = i + (0.16 if (i > 0 and dec_vals[i - 1] - dec > 30) else (0.08 if (i > 0 and dec_vals[i - 1] - dec > 18) else 0.0))
        y_lbl = dec + (9.5 if (i > 0 and dec_vals[i - 1] - dec > 30) else 7.5)
        axD.text(x_lbl, y_lbl, f"{dec:.1f}", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=TEXT)

    axD.set_xticks(x_idx)
    axD.set_xticklabels(tok_labels, fontsize=9.4, fontweight="bold", color=TEXT_SUB)
    axD.set_ylim(0, 248)
    axD.set_yticks([0, 50, 100, 150, 200])
    axD.set_ylabel("Decode (tok/s)", fontsize=10.4, fontweight="bold", color=TEXT_SUB)
    axD.tick_params(axis="y", colors=MUTED, labelsize=9.2)
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)
    axD.grid(axis="y", linestyle="--", alpha=0.18, color=MUTED)

    canvas.text(
        xL + 0.018,
        y2 + 0.022,
        "20/20 exact needle recall  ·  Prefill 5.8k → 2.4k tok/s  ·  40/48 looped layers use W=1024 SWA",
        fontsize=9.4,
        color=MUTED,
        va="bottom",
    )

    # === CARD E: What Changed (Clear, Scannable, Non-Fuzzy) ===
    _add_card(canvas, xR, y2, w_right, h2)
    canvas.text(xR + 0.018, y2 + h2 - 0.030, "WHAT CHANGED (SM86 PATCH)", fontsize=9.8, fontweight="bold", color=MUTED, family="DejaVu Sans Mono")
    canvas.text(xR + 0.018, y2 + h2 - 0.068, "3 fixes — zero C++ compile", fontsize=14.4, fontweight="bold", color=TEXT)

    store_exact_fmt = _fmt_ratio_str(kernel_json["store_exact_matches"])
    built_items = [
        (
            "1",
            CYAN,
            "#0d253a",
            "#1f5a85",
            "Native uint8 + RNE Store",
            f"Avoids Triton <SM89 fp8e4nv error.\n{store_exact_fmt} bit-exact store.",
        ),
        (
            "2",
            CYAN,
            "#0d253a",
            "#1f5a85",
            "5-Op SM86 PTX Dequant",
            f"Replaces SM89+ cvt.e4m3x2.\n{kernel_json['load_finite_exact']} finite E4M3 values exact.",
        ),
        (
            "3",
            EMERALD,
            "#0d2b2a",
            "#1d6b57",
            "DSpark Draft Guard Fix",
            "Keeps 3-layer drafter in BF16 KV;\nfixes upstream method='dspark' bug.",
        ),
    ]
    ny = y2 + h2 - 0.106
    for num, fg, bg, ec, title, desc in built_items:
        box_h = 0.082
        by = ny - box_h
        _add_card(canvas, xR + 0.016, by, w_right - 0.032, box_h, fc=CARD_ALT, ec=BORDER, lw=0.8, radius=0.008)
        _add_pill(canvas, xR + 0.025, by + box_h - 0.023, num, fg=fg, bg=bg, ec=ec, fontsize=9.6, pad=0.28)
        canvas.text(xR + 0.052, by + box_h - 0.023, title, fontsize=10.6, fontweight="bold", color=TEXT, va="center")
        canvas.text(
            xR + 0.052,
            by + box_h - 0.044,
            desc,
            fontsize=9.3,
            color=TEXT_SUB,
            va="top",
            linespacing=1.32,
        )
        ny = by - 0.011

    # Crisp bottom caveat line
    canvas.text(
        xR + 0.018,
        y2 + 0.020,
        f"Caveats: 32K prefill {pref_reg['speedup']:.2f}× · prior BF16 ref at 225W (FP8 250W)",
        fontsize=8.6,
        color=AMBER_SOFT,
        va="bottom",
    )

    canvas.text(
        0.5,
        0.019,
        "github.com/agent2x0r/inkling2-sm86-kernels   ·   Dockerfile · SM86 Patcher · Kernel Microbench · Raw JSONs",
        ha="center",
        va="center",
        fontsize=9.8,
        color=MUTED,
        family="DejaVu Sans Mono",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def _render_table_png(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    col_align: list[str],
    highlight_cols: set[int],
    footnote: str,
    out_path: Path,
    *,
    fig_w: float = 12.6,
    row_h: float = 0.56,
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    n_rows = len(rows)
    fig_h = 1.65 + (n_rows + 1) * row_h + 0.55
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=220, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer frame
    _add_card(ax, 0.008, 0.015, 0.984, 0.970, fc=BG, ec=BORDER, lw=1.0, radius=0.015)

    # Title
    top_pad = 0.88 if n_rows <= 6 else 0.91
    ax.text(0.035, top_pad, title, fontsize=15.0, fontweight="bold", color=TEXT, va="center")

    # Table geometry
    table_top = top_pad - 0.085
    table_bot = 0.145
    table_h = table_top - table_bot
    step_y = table_h / (n_rows + 1)
    left_x = 0.030
    total_w = 0.940

    # Header bar
    _add_card(ax, left_x, table_top - step_y, total_w, step_y, fc=HEADER_BG, ec="#314668", lw=1.0, radius=0.012)

    # Column x positions
    cum = [left_x + 0.015]
    usable_w = total_w - 0.030
    norm_w = [w / sum(col_widths) * usable_w for w in col_widths]
    for w in norm_w[:-1]:
        cum.append(cum[-1] + w)

    def _col_tx(j: int) -> float:
        al = col_align[j]
        if al == "left":
            gutter = 0.022 if (j > 0 and col_align[j - 1] == "right") else 0.0
            return cum[j] + gutter
        if al == "right":
            return cum[j] + norm_w[j] - 0.008
        return cum[j] + norm_w[j] / 2

    for j, h_text in enumerate(headers):
        al = col_align[j]
        ax.text(_col_tx(j), table_top - step_y / 2, h_text, fontsize=10.5, fontweight="bold", color=CYAN, va="center", ha=al)

    for i, row in enumerate(rows):
        ry_top = table_top - (i + 1) * step_y
        ry_bot = ry_top - step_y
        fc = ROW_EVEN if i % 2 == 1 else ROW_ODD
        _add_card(ax, left_x + 0.002, ry_bot + 0.004, total_w - 0.004, step_y - 0.008, fc=fc, ec="none", lw=0, radius=0.006)
        cy = (ry_top + ry_bot) / 2
        for j, cell in enumerate(row):
            al = col_align[j]
            is_hi = j in highlight_cols
            color = CYAN_SOFT if is_hi else (EMERALD if cell in ("PASS", "20/20", "0", "3/3", "1/1") else TEXT_SUB)
            weight = "bold" if (is_hi or j == 0 or color == EMERALD) else "normal"
            ax.text(_col_tx(j), cy, cell, fontsize=10.2, fontweight=weight, color=color, va="center", ha=al)

    ax.text(0.035, 0.068, footnote, fontsize=8.8, color=MUTED, va="center")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, facecolor=BG)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    bench_dir = Path(__file__).resolve().parent
    dspark_fp8 = json.loads((bench_dir / "results_dspark_fp8kv.json").read_text(encoding="utf-8"))
    longctx = json.loads((bench_dir / "results_longctx_fp8kv.json").read_text(encoding="utf-8"))
    acc_fp8 = json.loads((bench_dir / "results_accuracy_text_fp8kv.json").read_text(encoding="utf-8"))
    kernel_json = json.loads((bench_dir / "results_kernel_fp8kv.json").read_text(encoding="utf-8"))

    # 1. Hero Bento Card
    render_bento(
        dspark_fp8,
        longctx,
        acc_fp8,
        kernel_json,
        bench_dir / "inkling2_sm86_fp8kv_bento.png",
    )

    out_dir = bench_dir / "fp8kv_tables"

    # 2. Table 1: BF16 vs Software FP8 KV Capacity & Accuracy
    he = acc_fp8["benchmarks"]["humaneval"]
    gsm = acc_fp8["benchmarks"]["gsm8k"]
    c8 = [r for r in dspark_fp8["concurrency"] if r["concurrency"] == 8][0]["agg_decode_tps"]
    _render_table_png(
        "Dual RTX 3090 (SM86, TP=2, DSpark) — BF16 KV vs Software FP8 KV",
        ["Metric", "BF16 KV (auto)", "Software FP8 KV", "Delta / Notes"],
        [
            ["Unified KV Pool (0.98 util)", "239,061 tokens", "482,810 tokens", "2.02× capacity (8.26 GiB/GPU)"],
            ["Max Single-User Context", "212,992 (1.12×)", "425,984 (1.13×)", "2.00× max_model_len (416K)"],
            ["Hybrid Padding Layers", "18 layers (≤350% waste)", "0", "FP8 128×1B == BF16 draft 64×2B"],
            ["HumanEval pass@1 (164)", "160/164 (97.56%)", f"{he['passed']}/{he['total']} ({he['pass_at_1_pct']:.2f}%)", f"{he['agg_comp_tps']:.1f} tok/s agg (c=8)"],
            ["GSM8K Accuracy (200)", "196/200 (98.0%)", f"{gsm['passed']}/{gsm['total']} ({gsm['accuracy_pct']:.1f}%)", f"{gsm['agg_comp_tps']:.1f} tok/s agg (c=8)"],
            ["Needle Recall (0% hits)", "15/15 (1K–60K)", "20/20", "100% recall out to 336,567 tok"],
            ["Vision FA2 / Tool Calls", "2/2  ·  2/2", "2/2  ·  2/2", "Exact JSON args + FA2 parity"],
            ["c=8 Concurrency (4K/user)", "383.4 tok/s agg", f"{c8:.1f} tok/s agg", "Greedy 12-prompt: 239.3 tok/s"],
        ],
        [2.7, 2.1, 2.2, 3.6],
        ["left", "right", "right", "left"],
        {2},
        "Same dedicated 2× RTX 3090 (250W, 0.98 util) for KV pool stats. BF16 accuracy/concurrency refs from checked-in 225W baseline.",
        out_dir / "01_fp8kv_capacity_accuracy.png",
        row_h=0.52,
    )

    # 3. Table 2: Long-Context Scaling (1K -> 336.6K)
    ctx_rows = []
    for r in dspark_fp8["context_scaling"]:
        ctx_rows.append([
            r["bucket"],
            f"{r['prompt_tokens_mean']:,}",
            f"{r['ttft_s']['mean']:.2f}±{r['ttft_s']['std']:.2f}s",
            f"{r['prefill_tps']['mean']:,.0f}",
            f"{r['decode_tps']['mean']:.1f}±{r['decode_tps']['std']:.1f}",
            f"{r['spec_acc_len']['mean']:.2f}",
            str(r["prefix_cache_hits_total"]),
            r["needle_hits"],
        ])
    for r in longctx["long_context_fp8"]:
        ctx_rows.append([
            r["bucket"],
            f"{r['prompt_tokens']:,}",
            f"{r['ttft_s']:.2f}s",
            f"{r['prefill_tps']:,.0f}",
            f"{r['decode_tps']:.1f}",
            f"{r['spec_acc_len']:.2f}",
            str(r["prefix_cache_hits"]),
            "1/1" if r["needle_hit"] else "0/1",
        ])
    _render_table_png(
        "Long-Context Scaling — Software FP8 KV + DSpark (1K → 337K, 0% Prefix Hits)",
        ["Bucket", "Prompt Tok", "TTFT", "Prefill (tok/s)", "DSpark Dec (tok/s)", "Acc Len", "Prefix Hits", "NIAH"],
        ctx_rows,
        [1.0, 1.35, 1.45, 1.55, 1.95, 1.0, 1.15, 0.85],
        ["left", "right", "right", "right", "right", "right", "right", "center"],
        {4},
        "1K–60K: 3 trials ± 1σ (results_dspark_fp8kv.json). 64K–320K: uncached Python stdlib single-trial sweep (results_longctx_fp8kv.json). 20/20 needles PASS.",
        out_dir / "02_fp8kv_longctx_scaling.png",
        row_h=0.47,
    )

    # 4. Table 3: SM86 Kernel Microbenchmark
    k_rows = []
    for r in kernel_json["regimes"]:
        k_rows.append([
            r["regime"],
            f"{r['bf16_us']:,.1f} µs",
            f"{r['fp8_us']:,.1f} µs",
            f"{r['speedup']:.2f}×",
            f"{r['fp8_read_gbs']:.1f} GB/s" if r["q_len"] <= 8 else "compute-bound",
            f"{r['max_diff_vs_dequant_bf16']:.1e}",
        ])
    _render_table_png(
        "SM86 (RTX 3090) Attention Kernel — 5-Op Software E4M3 Dequant vs BF16 (L2-Flushed)",
        ["Attention Regime (H=12, HKV=4, D=128)", "BF16 KV", "SW FP8 KV", "Speedup", "FP8 Read BW", "Max Diff"],
        k_rows,
        [3.4, 1.3, 1.3, 1.1, 1.7, 1.2],
        ["left", "right", "right", "right", "right", "right"],
        {2, 3},
        f"Measured on {kernel_json['gpu']} via benchmarks/bench_fp8_kv_kernel.py. Store RNE exact: {_fmt_ratio_str(kernel_json['store_exact_matches'])}; Load finite exact: {kernel_json['load_finite_exact']}.",
        out_dir / "03_fp8kv_kernel_microbench.png",
        row_h=0.52,
    )


if __name__ == "__main__":
    main()
