# paths.py
import re
from pathlib import Path
from typing import Optional, Dict, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据集路径
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DATASET_DIR = DATASET_DIR / "Train"
TEST_DATASET_DIR = DATASET_DIR / "Test"
CHANNEL_PROBLEM_DIR = TEST_DATASET_DIR / "channel_problem"

# 检查点根目录 (新的统一目录结构)
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# 论文结果路径
PAPER_DIR = PROJECT_ROOT / "paper"
PAPER_RESULTS_DIR = PAPER_DIR / "results"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"

# 确保目录存在
for directory in [DATASET_DIR, TRAIN_DATASET_DIR, TEST_DATASET_DIR,
                  CHANNEL_PROBLEM_DIR, CHECKPOINTS_DIR,
                  PAPER_DIR, PAPER_RESULTS_DIR, PAPER_FIGURES_DIR]:
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


# ============================================================================
# 实验命名系统 - 版本树架构
# ============================================================================
#
# 命名格式：EXP_XX_NETTYPE_vYY_Description
#
# 示例：
#   EXP_01_RESNET_Base              # 第一个基础实验
#   EXP_02_LIGHTNET_Base            # 第二个基础实验
#   EXP_03_RESNET_v01_Pruning       # 基于 EXP_01 的剪枝实验
#   EXP_04_RESNET_v01_FineTune      # 基于 EXP_01 的微调实验
#   EXP_05_LIGHTNET_v02_KD          # 基于 EXP_02 的知识蒸馏
#   EXP_06_RESNET_v03_NewData       # 基于 EXP_03 的新数据实验
#
# 继承关系树：
#   EXP_01 (Base)
#   ├── EXP_03 (v01 - Pruning)
#   │   └── EXP_06 (v03 - NewData)
#   ├── EXP_04 (v01 - FineTune)
#   EXP_02 (Base)
#   └── EXP_05 (v02 - KD)
# ============================================================================


def get_next_experiment_number() -> int:
    """
    自动获取下一个实验序号

    Returns:
        int: 下一个实验序号 (从 1 开始)
    """
    if not CHECKPOINTS_DIR.exists():
        return 1

    # 查找所有符合 EXP_XX_* 格式的目录
    pattern = re.compile(r'^EXP_(\d+)_.*')
    max_num = 0

    for item in CHECKPOINTS_DIR.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

    return max_num + 1


def generate_experiment_name(
        net_type,
        description: str,
        base_version: Optional[str] = None
) -> str:
    """
    自动生成实验名称（支持版本继承关系）

    Args:
        net_type: 网络类型枚举 (如 NetworkType.ResNet)
        description: 实验描述 (如 "Base", "Pruning", "FineTune", "KD_PCA16")
        base_version: 基础版本号 (可选)
            - None: 表示这是一个独立的基础实验
            - "01": 表示基于 EXP_01 发展而来
            - "02": 表示基于 EXP_02 发展而来

    Returns:
        str: 完整的实验名称

    Examples:
        >>> # 创建基础实验
        >>> generate_experiment_name(NetworkType.ResNet, "Base")
        "EXP_01_RESNET_Base"

        >>> # 创建基于 EXP_01 的剪枝实验
        >>> generate_experiment_name(NetworkType.ResNet, "Pruning", base_version="01")
        "EXP_03_RESNET_v01_Pruning"

        >>> # 创建基于 EXP_01 的微调实验
        >>> generate_experiment_name(NetworkType.ResNet, "FineTune", base_version="01")
        "EXP_04_RESNET_v01_FineTune"

        >>> # 创建基于 EXP_02 的知识蒸馏实验
        >>> generate_experiment_name(NetworkType.LightNet, "KD_PCA16", base_version="02")
        "EXP_05_LIGHTNET_v02_KD_PCA16"
    """
    next_num = get_next_experiment_number()

    # 构建实验名称
    if base_version:
        # 扩展实验格式：EXP_XX_NETTYPE_vYY_Description
        exp_name = f"EXP_{next_num:02d}_{net_type.value}_v{base_version}_{description}"
    else:
        # 基础实验格式：EXP_XX_NETTYPE_Description
        exp_name = f"EXP_{next_num:02d}_{net_type.value}_{description}"

    return exp_name


class ExperimentInfo:
    """实验信息类，用于解析和存储实验元数据"""

    def __init__(self, experiment_name: str):
        """
        解析实验名称并提取元数据

        Args:
            experiment_name: 实验名称，如 "EXP_05_LIGHTNET_v02_KD_PCA16"
        """
        self.original_name = experiment_name
        self.exp_number: Optional[int] = None
        self.net_type: Optional[str] = None
        self.base_version: Optional[str] = None
        self.description: Optional[str] = None
        self.is_extension: bool = False

        self._parse_name(experiment_name)

    def _parse_name(self, experiment_name: str) -> None:
        """解析实验名称"""
        # 匹配扩展实验格式：EXP_XX_NETTYPE_vYY_Description
        pattern_ext = re.compile(r'^EXP_(\d+)_(.+?)_v(\d+)_(.+)$')
        # 匹配基础实验格式：EXP_XX_NETTYPE_Description
        pattern_base = re.compile(r'^EXP_(\d+)_(.+)_(.+)$')

        match = pattern_ext.match(experiment_name)
        if match:
            self.exp_number = int(match.group(1))
            self.net_type = match.group(2)
            self.base_version = match.group(3)
            self.description = match.group(4)
            self.is_extension = True
            return

        match = pattern_base.match(experiment_name)
        if match:
            self.exp_number = int(match.group(1))
            self.net_type = match.group(2)
            self.base_version = None
            self.description = match.group(3)
            self.is_extension = False
            return

        raise ValueError(f"无法解析实验名称：{experiment_name}")

    def get_base_experiment_number(self) -> Optional[int]:
        """获取基础实验编号"""
        if self.base_version:
            return int(self.base_version)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'exp_number': self.exp_number,
            'net_type': self.net_type,
            'base_version': self.base_version,
            'description': self.description,
            'is_extension': self.is_extension,
            'base_exp_number': self.get_base_experiment_number()
        }

    def __str__(self) -> str:
        """字符串表示"""
        if self.is_extension:
            return f"EXP_{self.exp_number:02d} ({self.net_type}) <- v{self.base_version}: {self.description}"
        else:
            return f"EXP_{self.exp_number:02d} ({self.net_type}): {self.description}"

    def __repr__(self) -> str:
        return f"ExperimentInfo('{self.original_name}')"


def parse_experiment_name(experiment_name: str) -> ExperimentInfo:
    """
    解析实验名称，返回 ExperimentInfo 对象

    Args:
        experiment_name: 实验名称

    Returns:
        ExperimentInfo: 包含实验元数据的对象

    Examples:
        >>> info = parse_experiment_name("EXP_05_LIGHTNET_v02_KD_PCA16")
        >>> info.exp_number
        5
        >>> info.net_type
        'LIGHTNET'
        >>> info.base_version
        '02'
        >>> info.is_extension
        True
        >>> info.get_base_experiment_number()
        2
    """
    return ExperimentInfo(experiment_name)


def get_base_experiment_dir(experiment_name: str) -> Optional[Path]:
    """
    根据实验名称获取其基础实验目录

    Args:
        experiment_name: 实验名称，如 "EXP_05_LIGHTNET_v02_KD"

    Returns:
        Path|None: 基础实验目录路径，如果该实验不是扩展实验则返回 None

    Examples:
        >>> get_base_experiment_dir("EXP_05_LIGHTNET_v02_KD")
        PosixPath('D:/.../checkpoints/experiments/EXP_02_...')

        >>> get_base_experiment_dir("EXP_01_RESNET_Base")
        None
    """
    try:
        exp_info = parse_experiment_name(experiment_name)
    except ValueError:
        return None

    if not exp_info.is_extension or not exp_info.base_version:
        return None

    # 查找基础实验目录
    base_num = exp_info.get_base_experiment_number()
    pattern = re.compile(f'^EXP_{base_num:02d}_.*')

    for item in CHECKPOINTS_DIR.iterdir():
        if item.is_dir() and pattern.match(item.name):
            return item

    return None


def get_experiment_lineage(experiment_name: str) -> list:
    """
    获取实验的完整继承链（从根到当前）

    Args:
        experiment_name: 实验名称

    Returns:
        list: 实验名称列表，按继承顺序排列

    Examples:
        >>> get_experiment_lineage("EXP_06_RESNET_v03_NewData")
        ['EXP_01_RESNET_Base', 'EXP_03_RESNET_v01_Pruning', 'EXP_06_RESNET_v03_NewData']
    """
    lineage = []
    current_name = experiment_name

    while True:
        lineage.insert(0, current_name)

        try:
            exp_info = parse_experiment_name(current_name)
        except ValueError:
            break

        if not exp_info.is_extension or not exp_info.base_version:
            break

        # 查找父实验
        base_num = exp_info.get_base_experiment_number()
        pattern = re.compile(f'^EXP_{base_num:02d}_.*')

        parent_found = False
        for item in CHECKPOINTS_DIR.iterdir():
            if item.is_dir() and pattern.match(item.name):
                current_name = item.name
                parent_found = True
                break

        if not parent_found:
            break

    return lineage


def get_experiment_dir(experiment_name):
    """
    获取实验检查点目录

    Args:
        experiment_name: 实验名称，如 "EXP_01_KD_PCA16"

    Returns:
        Path: 实验检查点目录路径

    Example:
        >>> get_experiment_dir("EXP_01_KD_PCA16")
        checkpoints/experiments/EXP_01_KD_PCA16
    """
    if experiment_name is not None:
        model_dir = CHECKPOINTS_DIR / experiment_name
        weights_dir = model_dir / "weights"
        eval_dir = model_dir / "eval"

        return model_dir, weights_dir, eval_dir
    else:
        return CHECKPOINTS_DIR

# 数据集文件路径
def get_dataset_path(key):
    """获取数据集路径的辅助函数"""
    if key in DATASET_FILES:
        return str(DATASET_FILES[key])
    raise ValueError(f"未知数据集：{key}")

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
