#!/usr/bin/env python3
"""Generate iterative_distillation.pdf — ResNet 3-round progressive compression."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
rounds = [1, 2, 3]

teacher_acc  = [95.93, 94.49, 85.43]
student_acc  = [94.27, 86.15, 47.12]
teacher_params = [203264,  13784,   3360]
student_params = [13784,    3360,   1756]
compression   = ["14.7×", "60.5×\n(cum.)", "115.8×\n(cum.)"]

# colors
C_TEACHER  = "#2c7bb6"   # blue
C_STUDENT  = "#d7191c"   # red
C_BAR      = "#fdae61"   # orange

# ── Figure ────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(5.5, 3.6))

# ── Left axis: Accuracy ───────────────────────────────────────────────
ax1.plot(rounds, teacher_acc, "o-", color=C_TEACHER, linewidth=2.0,
         markersize=9, markerfacecolor="white", markeredgewidth=2.0,
         label="Teacher Acc.")
ax1.plot(rounds, student_acc, "s-", color=C_STUDENT, linewidth=2.0,
         markersize=9, markerfacecolor="white", markeredgewidth=2.0,
         label="Student Acc.")

# accuracy drop annotations
offsets = [(0.15, -2.8), (0.15, -3.2), (0.15, -3.2)]
drops = ["−1.67%", "−8.35%", "−38.31% ✗"]
for i, (dx, dy) in enumerate(offsets):
    mid = (teacher_acc[i] + student_acc[i]) / 2
    ax1.annotate(drops[i], (rounds[i] + dx, mid),
                 fontsize=8, color=C_STUDENT if i < 2 else "#888888",
                 fontweight="bold")

ax1.set_ylabel("Accuracy (%)", fontsize=11)
ax1.set_ylim(40, 102)
ax1.set_xlim(0.7, 3.3)
ax1.set_xticks(rounds)
ax1.set_xlabel("Iteration Round", fontsize=11)

# ── Right axis: Parameters (log) ──────────────────────────────────────
ax2 = ax1.twinx()
bar_width = 0.22
bars = ax2.bar(np.array(rounds) - bar_width/2, student_params, bar_width,
               color=C_BAR, edgecolor="#cc7a20", linewidth=0.8, zorder=0,
               label="Student Params")

# param labels on bars
for i, (r, p) in enumerate(zip(rounds, student_params)):
    if p >= 1000:
        label = f"{p/1000:.1f}K"
    else:
        label = str(p)
    ax2.text(r, p + 0.05 * max(student_params), label,
             ha="center", va="bottom", fontsize=8, fontweight="bold")

# compression ratio labels
for i, (r, c) in enumerate(zip(rounds, compression)):
    ax2.text(r, student_params[i] * 0.35, c,
             ha="center", va="top", fontsize=7.5, color="#8B4513",
             fontweight="bold")

ax2.set_ylabel("Parameters (log scale)", fontsize=11)
ax2.set_yscale("log")
ax2.set_ylim(500, 500_000)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else str(int(x))))

# ── Threshold annotation ──────────────────────────────────────────────
ax1.axvline(x=3.4, ymin=0.52, ymax=0.75, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7)
ax1.annotate("Threshold\nδ = 20%", xy=(3.25, 68), fontsize=7.5,
             color="gray", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.8))

# ── Legend ────────────────────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                    loc="lower left", fontsize=8.5, framealpha=0.9,
                    ncol=1)
legend.set_zorder(10)

# ── Grid & Style ──────────────────────────────────────────────────────
ax1.grid(axis="y", alpha=0.35, linewidth=0.6)
ax1.set_axisbelow(True)

# ── Title ─────────────────────────────────────────────────────────────
ax1.set_title("Iterative Distillation on ResNet", fontsize=12,
              fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("iterative_distillation.pdf", dpi=150, bbox_inches="tight")
print("Saved: iterative_distillation.pdf")
