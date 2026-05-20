#!/usr/bin/env python3
"""Figure 5 — synthetic cohort performance.

Reads tsm_fno/results/tsm_80hz/{summary.json, stratified.txt} and produces:
  (a) RL2 on G across pressure, contrast, radius bins
  (b) Ring RL2 (epsilon) across pressure bins — the strain-head metric
  (c) Headline cohort metrics as text panel

Output: paper/figures/fig5_cohort.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RES  = REPO / "tsm_fno" / "results" / "tsm_80hz"


def parse_stratified(path: Path) -> dict[str, list[dict]]:
    """Parse the 'stratified.txt' tables into a {section: [rows]} dict."""
    sections: dict[str, list[dict]] = {}
    section = None
    rx = re.compile(
        r"^\s*([a-z]):\s*([\d\.\-]+)-([\d\.\-]+)\s+N=\s*(\d+)\s+"
        r"RL2_G=([\d\.]+)\s+RL2_eps=([\d\.\-eE\+]+)\s+ring_RL2=([\d\.]+)"
    )
    with open(path) as f:
        for line in f:
            if line.startswith("=="):
                section = line.strip("= \n")
                sections[section] = []
            elif section and (m := rx.match(line)):
                sections[section].append({
                    "key": m.group(1),
                    "lo":  float(m.group(2)),
                    "hi":  float(m.group(3)),
                    "n":   int(m.group(4)),
                    "rl2_G":   float(m.group(5)),
                    "rl2_eps": float(m.group(6)),
                    "ring_rl2": float(m.group(7)),
                })
    return sections


def main():
    out = Path(__file__).parent / "fig5_cohort.png"
    summary = json.load(open(RES / "summary.json"))
    strat   = parse_stratified(RES / "stratified.txt")

    by_p   = next((r for k, r in strat.items() if "pressure" in k.lower()), [])
    by_c   = next((r for k, r in strat.items() if "contrast" in k.lower()), [])
    by_a   = next((r for k, r in strat.items() if "radius"   in k.lower()), [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    # (a) RL2 on G across the three stratifications, side-by-side bars
    panel_a = axes[0]
    groups: list[tuple[str, list[dict]]] = [
        ("pressure [Pa]", by_p),
        ("contrast G_l/G_bg", by_c),
        ("radius [mm]", by_a),
    ]
    colors = ["steelblue", "seagreen", "tomato"]
    x_offset = 0
    xticks, xlabels = [], []
    for (label, rows), color in zip(groups, colors):
        if not rows:
            continue
        xs = np.arange(len(rows)) + x_offset
        ys = [r["rl2_G"] for r in rows]
        bar_labels = [f"{r['lo']:g}–{r['hi']:g}\n(n={r['n']})" for r in rows]
        panel_a.bar(xs, ys, color=color, alpha=0.85, edgecolor="black",
                    linewidth=0.5, label=label)
        xticks.extend(xs)
        xlabels.extend(bar_labels)
        x_offset += len(rows) + 1

    panel_a.set_xticks(xticks)
    panel_a.set_xticklabels(xlabels, fontsize=7)
    panel_a.set_ylabel(r"whole-field RL$^2$ on G")
    panel_a.set_title("(a) Stiffness error stratified by phantom mode")
    panel_a.axhline(0.10, color="black", linestyle=":", linewidth=1.0,
                     label="target < 0.10")
    panel_a.legend(loc="upper right", fontsize=8)
    panel_a.grid(axis="y", alpha=0.3)
    panel_a.set_ylim(0, max(0.12, max(r["rl2_G"] for r in by_p + by_c + by_a) * 1.2))

    # (b) Ring RL2 (epsilon) across pressure bins — the strain-head metric
    panel_b = axes[1]
    if by_p:
        xs = np.arange(len(by_p))
        ring = [r["ring_rl2"] for r in by_p]
        labels = [f"{r['lo']:g}–{r['hi']:g} Pa\n(n={r['n']})" for r in by_p]
        bars = panel_b.bar(xs, ring, color="darkorange", alpha=0.85,
                            edgecolor="black", linewidth=0.5)
        panel_b.set_xticks(xs)
        panel_b.set_xticklabels(labels, fontsize=9)
        for b, v in zip(bars, ring):
            panel_b.text(b.get_x() + b.get_width() / 2, v + 0.0005,
                          f"{v:.4f}", ha="center", fontsize=8)
    panel_b.set_ylabel(r"perilesional-shell RL$^2$ on $\varepsilon$")
    panel_b.set_title("(b) Strain-head error vs lesion pressure")
    panel_b.axhline(0.10, color="black", linestyle=":", linewidth=1.0,
                     label="target < 0.10")
    panel_b.legend(loc="upper right", fontsize=8)
    panel_b.grid(axis="y", alpha=0.3)
    panel_b.set_ylim(0, max(0.012, max(r["ring_rl2"] for r in by_p) * 2.5))

    # (c) Headline metrics text panel
    panel_c = axes[2]
    panel_c.axis("off")
    rows_txt = [
        ("validation samples", f"{summary.get('n_eval', '-'):,}"),
        ("RL² on G (whole)",   f"{summary.get('rl2_G_mean', float('nan')):.4f}"),
        ("ring RL² on ε",      f"{summary.get('ring_rl2_mean', float('nan')):.4f}"),
        ("SSIM on G",           f"{summary.get('ssim_G_mean', float('nan')):.4f}"),
        ("expansion AUC",       f"{summary.get('expansion_auc', float('nan')):.3f}"),
    ]
    panel_c.text(0.05, 0.95, "Headline metrics", fontsize=12, weight="bold",
                  transform=panel_c.transAxes)
    for i, (label, val) in enumerate(rows_txt):
        y = 0.82 - i * 0.13
        panel_c.text(0.05, y, label, fontsize=11, transform=panel_c.transAxes)
        panel_c.text(0.95, y, val, fontsize=11, transform=panel_c.transAxes,
                      ha="right", family="monospace", weight="bold")
    panel_c.set_title("(c) Held-out cohort summary")

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
