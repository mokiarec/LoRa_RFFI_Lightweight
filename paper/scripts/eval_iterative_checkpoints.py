#!/usr/bin/env python3
"""
Evaluate iterative-distillation checkpoints using the full multi-scenario
classification pipeline (PCA + KNN/SVM + sliding window voting),
identical to the protocol used in run_multi_clf_mode → test_classification.

Enrol:  residential file, dev 0-40, pkt 0-400
Test:   each channel scenario A-F, walk, mobile, antenna (dev 30-40, pkt 0-200)
"""
import sys
from collections import Counter
from pathlib import Path

_parent = Path(__file__).parent.parent
sys.path.insert(0, str(_parent))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pipeline.prune_builder import build_pruned_embedding_net
from dataset import DATASET
from core.config import PreprocessType, DEVICE
from net import NetworkType
from utils.data_preprocessor import load_generate, load_model as load_teacher_model

# ── Config ──────────────────────────────────────────────────────────
D_S = 8                       # PCA embedding dimension
VOTE_SIZE = 10
WEIGHT_KNN = 0.5
WEIGHT_SVM = 1.0 - WEIGHT_KNN
BATCH_SIZE = 128
PREPROCESS = PreprocessType.STFT

# Enrolment: always residential, devices 0-39, packets 0-399
ENROL_DEV = np.arange(0, 40, dtype=int)
ENROL_PKT = np.arange(0, 400, dtype=int)

# Iterative student checkpoints
CHECKPOINTS = {
    "Round 1": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_02/pruned/weights/Extractor_best.pth",
    "Round 2": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_03/pruned/weights/Extractor_best.pth",
    "Round 3": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_04/pruned/weights/Extractor_best.pth",
    "Round 4": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_05/pruned/weights/Extractor_best.pth",
    "Round 5": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_06/pruned/weights/Extractor_best.pth",
    "Round 6": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_07/pruned/weights/Extractor_best.pth",
    "Round 7": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_08/pruned/weights/Extractor_best.pth",
    "Round 8": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_09/pruned/weights/Extractor_best.pth",
    "Round 9": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_10/pruned/weights/Extractor_best.pth",

}

# Teacher checkpoint (from Round 1 iteration_01/large)
CHECKPOINTS_TEACHER = {
    "Teacher": _parent / "pipeline/runs_iterative/run_004_ResNet_iterative/iteration_01/large/weights/Extractor_best.pth",
}

# ── Scenario names matching test_multi_clf (DATASET['Channel']) ─────
SCENARIO_NAMES = [
    "LOS-A", "LOS-B", "LOS-C",
    "NLOS-D", "NLOS-E", "NLOS-F",
    "Walk-B", "Walk-F",
    "Mobile-Office", "Mobile-Meeting",
    "Antenna-B", "Antenna-F",
]
# Map display name -> channel key in DATASET['Channel']
SCENARIO_KEY = {
    "LOS-A": "A",         "LOS-B": "B",         "LOS-C": "C",
    "NLOS-D": "D",        "NLOS-E": "E",        "NLOS-F": "F",
    "Walk-B": "B_walk",   "Walk-F": "F_walk",
    "Mobile-Office": "moving_office",  "Mobile-Meeting": "moving_meeting_room",
    "Antenna-B": "B_antenna",          "Antenna-F": "F_antenna",
}


# ── Model loading ───────────────────────────────────────────────────

def load_pruned_model(ckpt_path):
    """Load a PrunableResNet student from pipeline checkpoint.

    Handles two formats:
      Format A (dict): {'state_dict': ..., 'channels': ..., 'embedding_dim': ...}
      Format B (OrderedDict): keys with 'embedding_net.' prefix, full-size model
    """
    saved = torch.load(ckpt_path, weights_only=True)

    # ── Format A: dict with metadata ──
    if isinstance(saved, dict) and 'state_dict' in saved:
        channels = saved['channels']
        embedding_dim = saved.get('embedding_dim', 8)
        emb_net = build_pruned_embedding_net("ResNet", 1, channels, embedding_dim)
        emb_net.load_state_dict(saved['state_dict'])
        return emb_net

    # ── Format B: raw OrderedDict with 'embedding_net.' prefix ──
    if isinstance(saved, dict):
        # Strip 'embedding_net.' prefix
        state_dict = {k.replace('embedding_net.', ''): v for k, v in saved.items()}
        # Infer channels from weight shapes
        conv1_out = state_dict['conv1.weight'].shape[0]
        l1_out    = state_dict['layer1.conv2.weight'].shape[0]
        l2_out    = state_dict['layer2.conv2.weight'].shape[0]
        l3_out    = state_dict['layer3.conv2.weight'].shape[0]
        l4_out    = state_dict['layer4.conv2.weight'].shape[0]
        channels = [conv1_out, l1_out, l2_out, l3_out, l4_out]
        embedding_dim = state_dict['fc.weight'].shape[0]
        emb_net = build_pruned_embedding_net("ResNet", 1, channels, embedding_dim)
        emb_net.load_state_dict(state_dict)
        return emb_net

    raise ValueError(f"Unknown checkpoint format: {type(saved)}")


def load_teacher_embedding(ckpt_path):
    """Load a standard TripletNet teacher and return its embedding_net."""
    model = load_teacher_model(ckpt_path, NetworkType.ResNet, generate_type=PREPROCESS)
    return model.embedding_net


# ── Feature extraction ──────────────────────────────────────────────

def extract_embeddings(embedding_net, data):
    """Batch-extract L2-normalized embeddings."""
    embeddings = []
    for i in range(0, len(data), BATCH_SIZE):
        batch = torch.tensor(data[i:i + BATCH_SIZE], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            emb = embedding_net(batch)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def sliding_vote(labels, vote_size):
    """Apply sliding window majority voting (matching test_classification)."""
    voted = []
    for i in range(len(labels)):
        start = max(0, i - vote_size // 2)
        end = min(len(labels), i + vote_size // 2 + 1)
        window = labels[start:end]
        voted.append(Counter(window).most_common(1)[0][0])
    return voted


# ── Full evaluation on a single model ───────────────────────────────

def evaluate_model(embedding_net):
    """
    Run the full multi-scenario classification pipeline on one embedding net.

    For each channel scenario:
      1. Load enrolment (residential) + test data
      2. Extract embeddings
      3. PCA(d=8) on enrolment → transform both
      4. KNN(k=5) + SVM(RBF) on enrolment PCA
      5. Predict test PCA → sliding window vote → combined weighted vote
      6. Record all accuracy metrics

    Returns:
      all_metrics: dict[scenario_name] -> {knn_wo, svm_wo, knn_w, svm_w, combined}
      total_params: int
    """
    embedding_net.to(DEVICE)
    embedding_net.eval()
    total_params = sum(p.numel() for p in embedding_net.parameters())

    enrol_info = DATASET['Test']['residential']
    enrol_labels, enrol_data = load_generate(
        enrol_info.path, ENROL_DEV, ENROL_PKT,
        PREPROCESS, snr_range=None)

    all_metrics = {}

    for scen_name in SCENARIO_NAMES:
        ch_key = SCENARIO_KEY[scen_name]
        ch_info = DATASET['Channel'][ch_key]

        print(f"\n  ── {scen_name} ({ch_info.note}) ──")

        # Load test data
        test_labels, test_data = load_generate(
            ch_info.path, ch_info.dev_range, ch_info.pkt_range,
            PREPROCESS, snr_range=None)

        # Extract embeddings
        enrol_emb = extract_embeddings(embedding_net, enrol_data)
        test_emb = extract_embeddings(embedding_net, test_data)
        print(f"  Enrol emb: {enrol_emb.shape}, Test emb: {test_emb.shape}")

        # PCA on enrolment
        pca = PCA(n_components=D_S)
        enrol_pca = pca.fit_transform(enrol_emb)
        test_pca = pca.transform(test_emb)

        # Classifiers
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        svm = SVC(kernel="rbf", C=1.0)
        knn.fit(enrol_pca, enrol_labels.ravel())
        svm.fit(enrol_pca, enrol_labels.ravel())

        # Predictions without voting
        pred_knn_wo = knn.predict(test_pca)
        pred_svm_wo = svm.predict(test_pca)

        # Sliding window voting
        pred_knn_w = sliding_vote(pred_knn_wo, VOTE_SIZE)
        pred_svm_w = sliding_vote(pred_svm_wo, VOTE_SIZE)

        # Combined weighted voting (matching test_classification)
        combined = []
        for i in range(0, len(pred_knn_w), VOTE_SIZE):
            window_end = min(i + VOTE_SIZE, len(pred_knn_w))
            knn_votes = Counter()
            svm_votes = Counter()
            for j in range(i, window_end):
                knn_votes[pred_knn_w[j]] += WEIGHT_KNN
                svm_votes[pred_svm_w[j]] += WEIGHT_SVM
            final_label = (knn_votes + svm_votes).most_common(1)[0][0]
            combined.extend([final_label] * (window_end - i))

        # Accuracies
        acc_knn_wo  = accuracy_score(test_labels, pred_knn_wo) * 100
        acc_svm_wo  = accuracy_score(test_labels, pred_svm_wo) * 100
        acc_knn_w   = accuracy_score(test_labels, pred_knn_w) * 100
        acc_svm_w   = accuracy_score(test_labels, pred_svm_w) * 100
        acc_combined = accuracy_score(test_labels, combined) * 100

        print(f"  KNN w/o vote: {acc_knn_wo:.2f}% | SVM w/o vote: {acc_svm_wo:.2f}%")
        print(f"  KNN w/  vote: {acc_knn_w:.2f}% | SVM w/  vote: {acc_svm_w:.2f}%")
        print(f"  Combined:     {acc_combined:.2f}%")

        all_metrics[scen_name] = {
            'knn_wo': acc_knn_wo,
            'svm_wo': acc_svm_wo,
            'knn_w':  acc_knn_w,
            'svm_w':  acc_svm_w,
            'combined': acc_combined,
        }

    return all_metrics, total_params


# ── Summary table ───────────────────────────────────────────────────

def print_summary_table(all_results):
    """Print formatted accuracy table and category averages."""
    print("\n" + "=" * 140)
    print("Summary: KNN w/ Sliding Voting Accuracy (%) Across All Scenarios")
    print("=" * 140)

    # Header
    print(f"{'Model':<22}", end="")
    for s in SCENARIO_NAMES:
        print(f"{s:>10}", end="")
    print()

    header_len = 22 + 10 * len(SCENARIO_NAMES)
    print("-" * header_len)

    for model_name in all_results:
        metrics = all_results[model_name]
        print(f"{model_name:<22}", end="")
        for s in SCENARIO_NAMES:
            acc = metrics.get(s, {}).get('knn_w', 0)
            print(f"{acc:10.2f}", end="")
        print()

    # Category averages
    print("\n--- Category Averages (KNN w/ Voting) ---")
    for model_name in all_results:
        r = all_results[model_name]
        los_avg    = np.mean([r[f"LOS-{c}"]['knn_w'] for c in "ABC"])
        nlos_avg   = np.mean([r[f"NLOS-{c}"]['knn_w'] for c in "DEF"])
        walk_avg   = np.mean([r["Walk-B"]['knn_w'], r["Walk-F"]['knn_w']])
        mobile_avg = np.mean([r["Mobile-Office"]['knn_w'],
                              r["Mobile-Meeting"]['knn_w']])
        ant_avg    = np.mean([r["Antenna-B"]['knn_w'],
                              r["Antenna-F"]['knn_w']])
        overall = np.mean([los_avg, nlos_avg, walk_avg, mobile_avg, ant_avg])
        print(f"  {model_name:<22}: "
              f"LOS={los_avg:.1f}  NLOS={nlos_avg:.1f}  "
              f"Walk={walk_avg:.1f}  Mobile={mobile_avg:.1f}  Ant={ant_avg:.1f}  "
              f"Overall={overall:.1f}")

    # Also print per-metric averages
    print("\n--- KNN wo/ Voting Averages ---")
    for model_name in all_results:
        r = all_results[model_name]
        avg = np.mean([r[s]['knn_wo'] for s in SCENARIO_NAMES])
        print(f"  {model_name:<22}: {avg:.1f}%")

    print("\n--- KNN w/ Voting Averages ---")
    for model_name in all_results:
        r = all_results[model_name]
        avg = np.mean([r[s]['knn_w'] for s in SCENARIO_NAMES])
        print(f"  {model_name:<22}: {avg:.1f}%")


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 140)
    print("Cross-Scenario Evaluation: Iterative Distillation Checkpoints")
    print("Full pipeline: PCA(8) + KNN(5) + SVM(RBF) + sliding window voting")
    print("=" * 140)

    all_results = {}

    # ── Evaluate teacher ──
    for name, ckpt in CHECKPOINTS_TEACHER.items():
        print(f"\n{'=' * 60}")
        print(f"Loading {name} ...")
        print(f"Path: {ckpt}")
        if not ckpt.exists():
            print(f"  ⚠️  Checkpoint not found, skipping.")
            continue
        emb_net = load_teacher_embedding(ckpt)
        n_params = sum(p.numel() for p in emb_net.parameters())
        print(f"  Params: {n_params:,}")
        metrics, _ = evaluate_model(emb_net)
        all_results[name] = metrics

    # ── Evaluate iterative students ──
    for name, ckpt in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"Loading {name} ...")
        print(f"Path: {ckpt}")
        if not ckpt.exists():
            print(f"  ⚠️  Checkpoint not found, skipping.")
            continue
        emb_net = load_pruned_model(ckpt)
        n_params = sum(p.numel() for p in emb_net.parameters())
        print(f"  Params: {n_params:,}")
        metrics, _ = evaluate_model(emb_net)
        all_results[name] = metrics

    # ── Print summary ──
    if all_results:
        print_summary_table(all_results)
    else:
        print("\n⚠️  No checkpoints were evaluated.")
