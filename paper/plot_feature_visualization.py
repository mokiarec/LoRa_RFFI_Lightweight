#!/usr/bin/env python3
"""
paper/plot_feature_visualization.py

Feature visualization across iterative distillation rounds on ResNet.
4 subplots left → right: Original → Student R1 → Student R2 → Student R3
Each subplot shows t-SNE projection of the embedding space, colored by device label.

Output: paper/figures/feature_visualization.pdf
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# ── Path setup ─────────────────────────────────────────────────────────
_parent = Path(__file__).parent.parent
sys.path.insert(0, str(_parent))

from core.config import PreprocessType, DEVICE
from net import NetworkType, TripletNet
from dataset import DATASET
from utils.data_preprocessor import load_generate_triplet
from pipeline.prune_builder import build_pruned_embedding_net

# ── Model paths ────────────────────────────────────────────────────────
PIPELINE_DIR = _parent / "pipeline"
RUNS_ITER = PIPELINE_DIR / "runs_iterative" / "run_004_ResNet_iterative"

# 4 models to visualize
MODEL_PATHS = [
    (RUNS_ITER / "iteration_01" / "pruned" / "weights" / "Extractor_best.pth", True,  "Teacher"),           # Teacher
    (RUNS_ITER / "iteration_02" / "pruned" / "weights" / "Extractor_best.pth", True,  "Student\nRound 1"),  # student R1
    (RUNS_ITER / "iteration_03" / "pruned" / "weights" / "Extractor_best.pth", True,  "Student\nRound 2"),  # student R2
    (RUNS_ITER / "iteration_04" / "pruned" / "weights" / "Extractor_best.pth", True, "Student\nRound 3"),  # student R3

]

# ── Output ─────────────────────────────────────────────────────────────
OUT_DIR = _parent / "paper" / "figures"
os.makedirs(str(OUT_DIR), exist_ok=True)
OUTPUT = OUT_DIR / "feature_visualization.pdf"

# ── Constants ──────────────────────────────────────────────────────────
PREPROCESS = PreprocessType.STFT
NET_TYPE = NetworkType.ResNet
MAX_DEVICES = 30         # max devices to show (colors)
MAX_SAMPLES_PER_DEV = 10  # samples per device
N_TSNE_ITER = 1000       # t-SNE iterations


def load_model(path, is_pruned):
    """Load a model (original or pruned format).

    Handles two pruned checkpoint formats:
      Format A (dict): {'state_dict': ..., 'channels': ..., 'embedding_dim': ...}
      Format B (OrderedDict): keys with 'embedding_net.' prefix, full-size model
    """
    saved = torch.load(str(path), map_location=DEVICE, weights_only=True)

    if is_pruned:
        # ── Format A: dict with metadata ──
        if isinstance(saved, dict) and 'state_dict' in saved:
            channels = saved.get("channels", None)
            embedding_dim = saved.get("embedding_dim", 8)
            net_type_str = saved.get("net_type", "ResNet")

            if channels is not None:
                emb_net = build_pruned_embedding_net(
                    net_type_str, PREPROCESS.in_channels, channels, embedding_dim)
                emb_net.load_state_dict(saved["state_dict"])
            else:
                raise ValueError("Pruned model missing 'channels' info")
        # ── Format B: raw OrderedDict with 'embedding_net.' prefix ──
        else:
            state_dict = {k.replace('embedding_net.', ''): v for k, v in saved.items()}
            conv1_out = state_dict['conv1.weight'].shape[0]
            l1_out    = state_dict['layer1.conv2.weight'].shape[0]
            l2_out    = state_dict['layer2.conv2.weight'].shape[0]
            l3_out    = state_dict['layer3.conv2.weight'].shape[0]
            l4_out    = state_dict['layer4.conv2.weight'].shape[0]
            channels = [conv1_out, l1_out, l2_out, l3_out, l4_out]
            embedding_dim = state_dict['fc.weight'].shape[0]
            emb_net = build_pruned_embedding_net(
                "ResNet", PREPROCESS.in_channels, channels, embedding_dim)
            emb_net.load_state_dict(state_dict)

        model = TripletNet(net_type=NET_TYPE, in_channels=PREPROCESS.in_channels)
        model.embedding_net = emb_net
    else:
        model = TripletNet(net_type=NET_TYPE, in_channels=PREPROCESS.in_channels)
        model.load_state_dict(saved)

    model = model.to(DEVICE)
    model.eval()
    return model


def extract_embeddings(model, data):
    """Extract embedding vectors from a batch of data."""
    model.eval()
    data = data.to(DEVICE)
    with torch.no_grad():
        embeddings = model.embedding_net(data)
    return embeddings.cpu().numpy()


def main():
    print("Loading test data...")
    file_path = str(DATASET["Test"]["seen"].path)
    label, triplet = load_generate_triplet(
        file_path, np.arange(0, 40), np.arange(0, 10),
        PREPROCESS, snr_range=None,
    )
    anchor_data = triplet[0]
    anchor_labels = label.numpy() if torch.is_tensor(label) else np.array(label)
    # Ensure 1D
    anchor_labels = np.squeeze(anchor_labels)
    print(f"Labels shape: {anchor_labels.shape}, unique: {len(np.unique(anchor_labels))}")

    # Subset: pick MAX_DEVICES devices, MAX_SAMPLES_PER_DEV each
    rng = np.random.RandomState(42)
    unique_devs = np.unique(anchor_labels)[:MAX_DEVICES]
    mask = np.isin(anchor_labels, unique_devs)
    sub_idx = np.where(mask)[0]
    # For each device, pick MAX_SAMPLES_PER_DEV
    keep = []
    for d in unique_devs:
        d_idx = np.where(anchor_labels == d)[0]
        d_idx = rng.choice(d_idx, min(MAX_SAMPLES_PER_DEV, len(d_idx)), replace=False)
        keep.extend(d_idx)
    keep = np.array(keep)
    keep.sort()

    data_subset = anchor_data[keep]
    labels_subset = anchor_labels[keep]
    print(f"Subset: {len(data_subset)} samples, {len(unique_devs)} devices × ≤{MAX_SAMPLES_PER_DEV}")

    # ── Load models & extract embeddings ───────────────────────────
    embeddings_list = []
    model_labels = []

    for path, is_pruned, label_name in MODEL_PATHS:
        if not path.exists():
            print(f"  SKIP (not found): {path}")
            continue
        print(f"Loading: {label_name}  ({'pruned' if is_pruned else 'original'})")
        model = load_model(path, is_pruned)
        emb = extract_embeddings(model, data_subset)
        embeddings_list.append(emb)
        model_labels.append(label_name)
        print(f"  Embedding dim: {emb.shape[1]}")

    # ── t-SNE on each embedding space (separate fit) ───────────────
    print("\nRunning t-SNE...")
    tsne_results = []
    for i, emb in enumerate(embeddings_list):
        print(f"  t-SNE on {model_labels[i]}  ({emb.shape[1]}-dim → 2-dim)")
        tsne = TSNE(n_components=2, perplexity=min(30, emb.shape[0] // 3),
                    random_state=42, max_iter=N_TSNE_ITER, n_jobs=1)
        emb_2d = tsne.fit_transform(emb)
        tsne_results.append(emb_2d)

    # ── Plot ───────────────────────────────────────────────────────
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    })

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))

    # IEEE-style high-contrast colors: Set1 (9) + Dark2 (8) + Set2 (8) for up to 25
    from matplotlib import colormaps as cm
    cmap1 = plt.get_cmap("Set1", 9)
    cmap2 = plt.get_cmap("Dark2", 8)
    cmap3 = plt.get_cmap("Set2", 8)
    colors = [cmap1(j) for j in range(9)] + [cmap2(j) for j in range(8)] + [cmap3(j) for j in range(8)]
    color_map = {d: colors[i % len(colors)] for i, d in enumerate(unique_devs)}

    for ax_idx, (ax, emb_2d, lbl) in enumerate(zip(axes, tsne_results, model_labels)):
        for dev in unique_devs:
            mask = (labels_subset == dev).ravel()  # ensure 1D
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[color_map[dev]], s=18, alpha=0.85,
                       edgecolors="none", label=f"Dev {dev}")

        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=6)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        # IEEE-style: bold frame
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")



    # ── Shared legend (top of figure) ──────────────────────────────
    legend_handles = []
    # Show a subset of devices in legend
    legend_step = max(1, len(unique_devs) // 12)
    for i, dev in enumerate(unique_devs):
        if i % legend_step == 0:
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[dev], markersize=8,
                           label=f"Dev {dev}")
            )
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(15, len(legend_handles)),
               fontsize=10, framealpha=0.9, edgecolor="#cccccc",
               markerscale=2.0, mode="expand", columnspacing=0.6,
               bbox_to_anchor=(0, 1.02, 1, 0), handletextpad=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(str(OUTPUT), dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
