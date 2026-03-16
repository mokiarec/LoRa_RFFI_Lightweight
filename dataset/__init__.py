# dataset/__init__.py
"""
数据集说明文档

本模块定义了项目中使用的各种数据集的元数据和配置信息。
所有数据集信息集中管理，便于维护和统一调用。
"""
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# 数据集目录
DATASET_ROOT = Path(__file__).parent.absolute()

# 数据集路径
TRAIN_DATASET_DIR = DATASET_ROOT / "Train"
TEST_DATASET_DIR = DATASET_ROOT / "Test"
CHANNEL_PROBLEM_DIR = TEST_DATASET_DIR / "channel_problem"

# 确保目录存在
for directory in [TRAIN_DATASET_DIR, TEST_DATASET_DIR, CHANNEL_PROBLEM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetInfo:
    """数据集信息类"""
    name: str  # 用于显示的名称
    path: Path  # H5 文件路径
    dev_range: np.ndarray = None  # 设备范围
    pkt_range: np.ndarray = None  # 数据包范围
    note: str = ""  # 备注信息（场景描述）


# ============================================================================
# 数据集定义
# ============================================================================
# 这里只定义数据集的元数据信息
# ============================================================================

DATASET = {
    # ==================== 训练集 ====================
    'train_no_aug': DatasetInfo(
        name="Train No Aug",
        path=TRAIN_DATASET_DIR / "dataset_training_no_aug.h5",
        note="无增强训练数据"
    ),

    'train_aug': DatasetInfo(
        name="Train Aug",
        path=TRAIN_DATASET_DIR / "dataset_training_aug.h5",
        note="带增强训练数据"
    ),

    'train_aug_0hz': DatasetInfo(
        name="Train Aug 0Hz",
        path=TRAIN_DATASET_DIR / "dataset_training_aug_0hz.h5",
        note="0Hz 频偏增强训练数据"
    ),

    # ==================== 基础测试集 ====================
    'test_seen': DatasetInfo(
        name="Seen Devices",
        path=TEST_DATASET_DIR / "dataset_seen_devices.h5",
        dev_range=np.arange(0, 30),
        pkt_range=np.arange(0, 100),
        note="Residential, LOS, stationary"
    ),

    'test_residential': DatasetInfo(
        name="Residential Enrol",
        path=TEST_DATASET_DIR / "dataset_residential.h5",
        dev_range=np.arange(0, 30),
        pkt_range=np.arange(0, 100),
        note="住宅环境注册数据"
    ),

    'test_rogue': DatasetInfo(
        name="Rogue Device Detection",
        path=TEST_DATASET_DIR / "dataset_rogue.h5",
        dev_range=np.arange(0, 30),
        pkt_range=np.arange(0, 100),
        note="Residential, NLOS, stationary - 用于恶意设备检测"
    ),

    # ==================== 信道场景 A-F (Channel Problem) ====================
    'A': DatasetInfo(
        name="Channel A (LOS)",
        path=CHANNEL_PROBLEM_DIR / "A.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="LOS, stationary"
    ),

    'B': DatasetInfo(
        name="Channel B (LOS)",
        path=CHANNEL_PROBLEM_DIR / "B.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="LOS, stationary"
    ),

    'C': DatasetInfo(
        name="Channel C (LOS)",
        path=CHANNEL_PROBLEM_DIR / "C.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="LOS, stationary"
    ),

    'D': DatasetInfo(
        name="Channel D (NLOS)",
        path=CHANNEL_PROBLEM_DIR / "D.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="NLOS, stationary"
    ),

    'E': DatasetInfo(
        name="Channel E (NLOS)",
        path=CHANNEL_PROBLEM_DIR / "E.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="NLOS, stationary"
    ),

    'F': DatasetInfo(
        name="Channel F (NLOS)",
        path=CHANNEL_PROBLEM_DIR / "F.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="NLOS, stationary"
    ),

    # ==================== 移动性与天线 (Mobility & Antenna) ====================
    'B_walk': DatasetInfo(
        name="B walk (LOS)",
        path=CHANNEL_PROBLEM_DIR / "B_walk.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Object moving"
    ),

    'F_walk': DatasetInfo(
        name="F walk (NLOS)",
        path=CHANNEL_PROBLEM_DIR / "F_walk.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Object moving"
    ),

    'moving_office': DatasetInfo(
        name="Moving Office",
        path=CHANNEL_PROBLEM_DIR / "moving_office.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Mobile"
    ),

    'moving_meeting_room': DatasetInfo(
        name="Moving Meeting Room",
        path=CHANNEL_PROBLEM_DIR / "moving_meeting_room.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Mobile, NLOS"
    ),

    'B_antenna': DatasetInfo(
        name="B antenna",
        path=CHANNEL_PROBLEM_DIR / "B_antenna.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Parallel antenna"
    ),

    'F_antenna': DatasetInfo(
        name="F antenna",
        path=CHANNEL_PROBLEM_DIR / "F_antenna.h5",
        dev_range=np.arange(30, 40),
        pkt_range=np.arange(0, 200),
        note="Parallel antenna, NLOS"
    ),
}