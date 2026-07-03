#!/usr/bin/env python3
"""
paper/plot_iterative_pca_per_layer.py

Per-layer PCA effective dimensions across iterative distillation rounds on ResNet.

Layout:
  Top:    Accuracy line chart (teacher vs student across rounds)
  Bottom: Per-layer effective channel dimension evolution across rounds
          (layers on y-axis, showing channel count at each point)

Output: paper/figures/iterative_distillation.pdf
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────
_parent = Path(__file__).parent.parent
sys.path.insert(0, str(_parent))

BASE_DIR = _parent / "pipeline" / "runs_iterative" / "run_001_ResNet_iterative"

ITER_PCA = {
    1: BASE_DIR / "iteration_01" / "pca_results.json",
    2: BASE_DIR / "iteration_02" / "pca_results.json",
    3: BASE_DIR / "iteration_03" / "pca_results.json",
}
ITER_SUMMARY = BASE_DIR / "iterative_results.json"

OUT_DIR = _parent / "paper" / "figures"
os.makedirs(str(OUT_DIR), exist_ok=True)
OUTPUT = OUT_DIR / "iterative_distillation.pdf"


# ── Styling ─────────────────────────────────────────────────────────────
C_TEACHER = "#2c7bb6"   # blue
C_STUDENT = "#d7191c"   # red

# Per-stage colors for bottom chart
STAGE_COLORS = {
    "conv1":  "#4c72b0",
    "layer1": "#55a868",
    "layer2": "#c44e52",
    "layer3": "#8172b2",
    "layer4": "#ccb974",
}
STAGE_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4"]


def short_name(name: str) -> str:
    """Simplify layer name: embedding_net.layer2.conv1 -> L2.C1"""
    s = name.replace("embedding_net.", "")
    s = s.replace("conv1", "C1").replace("conv2", "C2")
    s = s.replace("shortcut", "SC")
    s = s.replace("layer", "L")
    return s


def stage_of(name: str) -> str:
    for s in STAGE_ORDER:
        if s in name:
            return s
    return "other"


def load():
    pca = {}
    for r, path in ITER_PCA.items():
        with open(path) as f:
            pca[r] = json.load(f)
    with open(ITER_SUMMARY) as f:
        summary = json.load(f)
    return pca, summary


def fmt(n):
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


def plot(pca, summary):
    # ── Font ────────────────────────────────────────────────────────────
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })

    rounds_list = [1, 2, 3]
    rounds_data = summary["results"]

    # ── Figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 7.5))

    # ============ TOP: Accuracy line chart ==============================
    ax_acc = fig.add_axes([0.08, 0.56, 0.38, 0.38])

    teacher_acc = [r["teacher_accuracy"] for r in rounds_data]
    student_acc = [r["student_accuracy"] for r in rounds_data]

    ax_acc.plot(rounds_list, teacher_acc, "o-", color=C_TEACHER, linewidth=2.2,
                markersize=9, markerfacecolor="white", markeredgewidth=2.0,
                label="Teacher Acc.")
    ax_acc.plot(rounds_list, student_acc, "s-", color=C_STUDENT, linewidth=2.2,
                markersize=9, markerfacecolor="white", markeredgewidth=2.0,
                label="Student Acc.")

    # Annotate accuracy values
    for i, (t, s) in enumerate(zip(teacher_acc, student_acc)):
        ax_acc.annotate(f"{t:.1f}", (rounds_list[i], t),
                        textcoords="offset points", xytext=(4, 7),
                        fontsize=8, color=C_TEACHER, fontweight="bold")
        ax_acc.annotate(f"{s:.1f}", (rounds_list[i], s),
                        textcoords="offset points", xytext=(4, -12),
                        fontsize=8, color=C_STUDENT, fontweight="bold")

    # Drop annotations
    for i, r in enumerate(rounds_data):
        drop = r["accuracy_drop"]
        mid = (r["teacher_accuracy"] + r["student_accuracy"]) / 2
        color = C_STUDENT if drop < 20 else "#999999"
        label = f"Δ = −{drop:.1f}%"
        if drop > 20:
            label += " [FAIL]"
        ax_acc.annotate(label, (rounds_list[i] + 0.12, mid),
                        fontsize=7.8, color=color, fontweight="bold",
                        va="center")

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xlabel("Iteration Round", fontsize=11)
    ax_acc.set_xticks(rounds_list)
    ax_acc.set_ylim(35, 102)
    ax_acc.set_xlim(0.7, 3.3)
    ax_acc.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax_acc.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax_acc.set_axisbelow(True)
    ax_acc.set_title("(a) Iterative Distillation Accuracy", fontsize=12,
                     fontweight="bold", pad=10)

    # ============ TOP-RIGHT: Params summary =============================
    ax_param = fig.add_axes([0.54, 0.56, 0.42, 0.38])
    ax_param.axis("off")
    ax_param.set_title("(b) Parameter & Compression Summary", fontsize=12,
                       fontweight="bold", pad=10)

    col_lbl = ["Round 1", "Round 2", "Round 3"]
    rows = [
        ["Teacher Params", *[fmt(r["teacher_params"]) for r in rounds_data]],
        ["Student Params", *[fmt(r["student_params"]) for r in rounds_data]],
        ["Compression",  "14.7×", "4.1× (60.5× cum.)", "1.9× (115.8× cum.)"],
        ["Acc Drop",       "−1.67%", "−8.35%", "−38.31% [FAIL]"],
    ]
    row_lbl = [r[0] for r in rows]
    cell_text = [r[1:] for r in rows]

    tbl = ax_param.table(cellText=cell_text, rowLabels=row_lbl, colLabels=col_lbl,
                         cellLoc="center", rowLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(0.95, 2.0)

    for key, cell in tbl.get_celld().items():
        cell.set_linewidth(0.6)
        if key[0] == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif key[1] == -1:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold", ha="left")
        # Highlight FAIL row
        if key[0] == 4 and key[1] == 3:
            cell.set_facecolor("#ffe0e0")
            cell.set_text_props(color="#cc0000", fontweight="bold")
        # Highlight compression row
        if key[0] == 3:
            cell.set_text_props(fontweight="bold")

    # ============ BOTTOM: Per-layer effective dim evolution =============
    ax_bot = fig.add_axes([0.08, 0.06, 0.88, 0.44])

    # Collect all unique layers across rounds in stage order
    all_layers = []
    seen = set()
    for r in [1, 2, 3]:
        for name in pca[r]:
            if name not in seen:
                all_layers.append(name)
                seen.add(name)
    # Sort by stage
    all_layers.sort(key=lambda n: (STAGE_ORDER.index(stage_of(n))
                                    if stage_of(n) in STAGE_ORDER else 99, n))

    n_layers = len(all_layers)
    y_pos = np.arange(n_layers)

    # Plot each layer as a horizontal line of points across rounds
    for li, layer_name in enumerate(all_layers):
        stg = stage_of(layer_name)
        color = STAGE_COLORS.get(stg, "#888888")

        vals = []
        xs = []
        for r in [1, 2, 3]:
            if layer_name in pca[r]:
                vals.append(pca[r][layer_name]["effective_dim"])
                xs.append(r)
            else:
                vals.append(np.nan)
                xs.append(r)

        # Filter out NaN for plotting
        valid = [(x, v) for x, v in zip(xs, vals) if not np.isnan(v)]
        if len(valid) < 2:
            # Single point or none — just plot markers
            for x, v in valid:
                ax_bot.plot(x, li, "o", color=color, markersize=8,
                            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
                ax_bot.annotate(str(v), (x, li), textcoords="offset points",
                                xytext=(10, 0), fontsize=7.5, color=color,
                                fontweight="bold", va="center")
            continue

        vx = [p[0] for p in valid]
        vy = [p[1] for p in valid]

        # Draw connecting line
        ax_bot.plot(vx, [li]*len(vx), "-", color=color, linewidth=1.8,
                    alpha=0.35, zorder=2)

        # Draw markers + value labels
        for x, v in zip(xs, vals):
            if np.isnan(v):
                # Mark as removed with faint X
                ax_bot.plot(x, li, "x", color="#cccccc", markersize=7,
                            markeredgewidth=1.2, zorder=3)
                continue
            # Circle marker
            ax_bot.plot(x, li, "o", color=color, markersize=9,
                        markerfacecolor="white", markeredgewidth=2.0,
                        zorder=5)
            # Value label
            ax_bot.annotate(str(v), (x, li),
                            textcoords="offset points",
                            xytext=(11, 1), fontsize=7.5, color=color,
                            fontweight="bold", va="center")

    # Also connect across rounds for same layer (trend line)
    for li, layer_name in enumerate(all_layers):
        stg = stage_of(layer_name)
        color = STAGE_COLORS.get(stg, "#888888")

        xs_conn = []
        for r in [1, 2, 3]:
            if layer_name in pca[r]:
                xs_conn.append(r)
        if len(xs_conn) >= 2:
            # Plot a subtle step-like trend
            vals_conn = [pca[r][layer_name]["effective_dim"] for r in xs_conn]
            # Normalize to y-offset for visual trend (small multiples)
            norm_vals = [li + (v - min(vals_conn)) / (max(vals_conn) - min(vals_conn) + 1) * 0.4
                         for v in vals_conn]
            ax_bot.plot(xs_conn, [li]*len(xs_conn), "-", color=color,
                        linewidth=2.5, alpha=0.8, zorder=1)

    # Y-axis: layer names
    labels = [short_name(n) for n in all_layers]
    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(labels, fontsize=7.5)
    ax_bot.set_xlabel("Iteration Round", fontsize=11)
    ax_bot.set_xticks([1, 2, 3])
    ax_bot.set_xlim(0.6, 3.4)
    ax_bot.set_ylim(-0.8, n_layers - 0.2)
    ax_bot.invert_yaxis()
    ax_bot.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax_bot.grid(axis="y", alpha=0.15, linewidth=0.4)
    ax_bot.set_axisbelow(True)

    # Stage separators on the right
    last_stage = None
    for li, layer_name in enumerate(all_layers):
        stg = stage_of(layer_name)
        if stg != last_stage and stg in STAGE_COLORS:
            ax_bot.annotate(stg.replace("layer", "Stage ").replace("conv1", "Stem"),
                            xy=(3.42, li), fontsize=7, color=STAGE_COLORS[stg],
                            fontweight="bold", va="center", ha="left",
                            fontstyle="italic")
            last_stage = stg

    ax_bot.set_title("(c) Per-Layer Effective Channels Across Iterations",
                     fontsize=12, fontweight="bold", pad=8)

    # ── Legend for bottom chart ────────────────────────────────────────
    legend_handles = []
    for stg in STAGE_ORDER:
        c = STAGE_COLORS[stg]
        lbl = stg.replace("conv1", "Stem").replace("layer", "Stage ")
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color=c, linewidth=2,
                       markersize=7, markerfacecolor="white",
                       markeredgewidth=2.0, label=lbl)
        )
    legend_handles.append(
        plt.Line2D([0], [0], marker="x", color="#cccccc", linewidth=0,
                   markersize=7, markeredgewidth=1.2, label="Removed")
    )
    ax_bot.legend(handles=legend_handles, loc="upper right",
                  fontsize=7.5, framealpha=0.9, ncol=3)

    # ── Save ────────────────────────────────────────────────────────────
    plt.savefig(str(OUTPUT), dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.close()


def main():
    if not BASE_DIR.exists():
        print(f"ERROR: {BASE_DIR} not found"); sys.exit(1)
    pca, summary = load()
    print(f"Loaded {len(pca)} rounds, {summary['total_iterations']} iterations")
    plot(pca, summary)


if __name__ == "__main__":
    main()
