# \modes\multi_dataset_classification_mode.py
"""多数据集分类评估模式相关函数"""
import numpy as np

from core.config import Config
from dataset import DATASET
from modes.classification_mode import test_classification


def test_multi_clf(
        config: Config,
):
    """
    在多个不同数据集上进行分类评估

    :param config: 配置对象
    """

    print(f"\n{'=' * 60}")
    print("多数据集分类评估")
    print(f"{'=' * 60}\n")

    for key, info in DATASET['Channel'].items():
        name = info.name
        path = info.path
        dev_range = info.dev_range
        pkt_range = info.pkt_range
        note = info.note

        print(f"\n{'=' * 60}")
        print(f"=== Evaluating: {name} ===")
        print(f"Note: {note}")
        print(f"{'=' * 60}\n")

        file_path_enrol=DATASET['Test']['residential'].path
        dev_range_enrol=np.arange(0, 40, dtype=int)
        pkt_range_enrol=np.arange(0, 400, dtype=int)

        # 执行分类测试
        test_classification(
            config=config,
            dataset_name=name,
            file_path_enrol=file_path_enrol,
            file_path_clf=path,
            dev_range_enrol=dev_range_enrol,
            pkt_range_enrol=pkt_range_enrol,
            dev_range_clf=dev_range,
            pkt_range_clf=pkt_range,
            is_pac=config.is_pca_test,
            enable_plots=False,
        )
