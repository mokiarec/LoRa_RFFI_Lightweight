# paths.py
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据集路径
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DATASET_DIR = DATASET_DIR / "Train"
TEST_DATASET_DIR = DATASET_DIR / "Test"
CHANNEL_PROBLEM_DIR = TEST_DATASET_DIR / "channel_problem"

# 模型根路径
ROOT_MODEL_DIR = PROJECT_ROOT / "model"

# 论文结果路径
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_RESULTS_DIR = PAPER_DIR / "results"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"

# 确保目录存在
for directory in [DATASET_DIR, TRAIN_DATASET_DIR, TEST_DATASET_DIR,
                  CHANNEL_PROBLEM_DIR, ROOT_MODEL_DIR, PAPER_DIR,
                  PAPER_RESULTS_DIR, PAPER_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 数据集文件路径
DATASET_FILES = {
    # Training datasets
    'train_no_aug': TRAIN_DATASET_DIR / "dataset_training_no_aug.h5",
    'train_aug_0hz': TRAIN_DATASET_DIR / "dataset_training_aug_0hz.h5",
    'train_aug': TRAIN_DATASET_DIR / "dataset_training_aug.h5",
    # Test datasets
    'test_seen': TEST_DATASET_DIR / "dataset_seen_devices.h5",
    'test_rogue': TEST_DATASET_DIR / "dataset_rogue.h5",
    'test_residential': TEST_DATASET_DIR / "dataset_residential.h5",
    # Channel problem datasets
    'A': CHANNEL_PROBLEM_DIR / "A.h5",
    'B': CHANNEL_PROBLEM_DIR / "B.h5",
    'C': CHANNEL_PROBLEM_DIR / "C.h5",
    'D': CHANNEL_PROBLEM_DIR / "D.h5",
    'E': CHANNEL_PROBLEM_DIR / "E.h5",
    'F': CHANNEL_PROBLEM_DIR / "F.h5",
    'B_walk': CHANNEL_PROBLEM_DIR / "B_walk.h5",
    'F_walk': CHANNEL_PROBLEM_DIR / "F_walk.h5",
    'moving_office': CHANNEL_PROBLEM_DIR / "moving_office.h5",
    'moving_meeting_room': CHANNEL_PROBLEM_DIR / "moving_meeting_room.h5",
    'B_antenna': CHANNEL_PROBLEM_DIR / "B_antenna.h5",
    'F_antenna': CHANNEL_PROBLEM_DIR / "F_antenna.h5",
}

# 模型文件路径
def get_model_path(preprocess_type, net_type, filename="", wst_j=None, wst_q=None):
    """获取模型路径"""
    if wst_j is not None and wst_q is not None:
        return ROOT_MODEL_DIR / f'preprocess_type.value_j{wst_j}q{wst_q}' / net_type.value / filename
    else:
        return ROOT_MODEL_DIR / preprocess_type.value / net_type.value / filename
# 数据集文件路径
def get_dataset_path(key):
    """获取数据集路径的辅助函数"""
    if key in DATASET_FILES:
        return str(DATASET_FILES[key])

# 论文输出文件路径
PAPER_OUTPUT_FILES = {
    'model_statistics': PAPER_RESULTS_DIR / "model_statistics.csv",
    'channel_robustness': PAPER_FIGURES_DIR / "channel_robustness_comparison.pdf",
    'mobility_robustness': PAPER_FIGURES_DIR / "mobility_robustness.pdf",
    'accuracy_heatmap': PAPER_FIGURES_DIR / "accuracy_heatmap_conditions.pdf",
    'detailed_accuracy_heatmap': PAPER_FIGURES_DIR / "detailed_accuracy_heatmap.pdf",
    'pca_ablation': PAPER_FIGURES_DIR / "pca_ablation_scenario_based.pdf",
    'pca_origin_pca': PAPER_FIGURES_DIR / "comparison_origin_pca.pdf",
}
