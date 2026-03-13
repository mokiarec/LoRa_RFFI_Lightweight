# paper/analysis_multi_dataset_classification.py
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, Mode, NetworkType, DistillateMode
from modes.classification_mode import test_classification
from paths import DATASET_FILES, get_model_path

# ========= 测试数据集定义 =========
test_datasets = [
    # name, path, dev_range, pkt_range, note
    ("Seen Devices",
     str(DATASET_FILES['test_seen']),
     np.arange(0, 30), np.arange(0, 100),
     "Residential, LOS, stationary"),

    # Channel A–F
    ("Channel A (LOS)",
     str(DATASET_FILES['A']),
     np.arange(30, 40), np.arange(0, 200),
     "LOS, stationary"),

    ("Channel B (LOS)",
     str(DATASET_FILES['B']),
     np.arange(30, 40), np.arange(0, 200),
     "LOS, stationary"),

    ("Channel C (LOS)",
     str(DATASET_FILES['C']),
     np.arange(30, 40), np.arange(0, 200),
     "LOS, stationary"),

    ("Channel D (NLOS)",
     str(DATASET_FILES['D']),
     np.arange(30, 40), np.arange(0, 200),
     "NLOS, stationary"),

    ("Channel E (NLOS)",
     str(DATASET_FILES['E']),
     np.arange(30, 40), np.arange(0, 200),
     "NLOS, stationary"),

    ("Channel F (NLOS)",
     str(DATASET_FILES['F']),
     np.arange(30, 40), np.arange(0, 200),
     "NLOS, stationary"),

    # Mobility
    ("B walk (LOS)",
     str(DATASET_FILES['B_walk']),
     np.arange(30, 40), np.arange(0, 200),
     "Object moving"),

    ("F walk (NLOS)",
     str(DATASET_FILES['F_walk']),
     np.arange(30, 40), np.arange(0, 200),
     "Object moving"),

    ("Moving Office",
     str(DATASET_FILES['moving_office']),
     np.arange(30, 40), np.arange(0, 200),
     "Mobile"),

    ("Moving Meeting Room",
     str(DATASET_FILES['moving_meeting_room']),
     np.arange(30, 40), np.arange(0, 200),
     "Mobile, NLOS"),

    # Antenna mismatch
    ("B antenna",
     str(DATASET_FILES['B_antenna']),
     np.arange(30, 40), np.arange(0, 200),
     "Parallel antenna"),

    ("F antenna",
     str(DATASET_FILES['F_antenna']),
     np.arange(30, 40), np.arange(0, 200),
     "Parallel antenna, NLOS"),
]

def run_multi_dataset_eval():
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.DISTILLATION
    config = Config(Mode.CLASSIFICATION,
         net_type=NetworkType.RESNET,
         teacher_net_type=NetworkType.RESNET,
         student_net_type=NetworkType.MobileNetV1,
         distillate_mode=DistillateMode.ONLY_TEST,
         test_list=[200],
         is_pca_test=False)

    results = []

    for name, path, dev_range, pkt_range, note in test_datasets:
        print(f"\n=== Evaluating: {name} ===")

        # 根据数据集名称决定使用的注册集文件
        if "SEEN" in name.upper():
            enrol_cfg = dict(
                file_path_enrol=str(DATASET_FILES['train_no_aug']),
                dev_range_enrol=np.arange(0, 40, dtype=int),
                pkt_range_enrol=np.arange(0, 200, dtype=int),
            )
        else:
            enrol_cfg = dict(
                file_path_enrol=str(DATASET_FILES['test_residential']),
                dev_range_enrol=np.arange(0, 40, dtype=int),
                pkt_range_enrol=np.arange(0, 400, dtype=int),
            )

        acc = test_classification(
            mode=config.mode,
            file_path_clf=path,
            dev_range_clf=dev_range,
            pkt_range_clf=pkt_range,
            net_type=config.NET_TYPE if config.mode == Mode.CLASSIFICATION else config.STUDENT_NET_TYPE,
            preprocess_type=config.PREPROCESS_TYPE,
            test_list=config.TEST_LIST,
            model_dir=config.ORIGIN_MODEL_DIR_PATH if config.mode == Mode.CLASSIFICATION else config.STUDENT_MODEL_DIR,
            is_pac=config.IS_PCA_TEST,
            **enrol_cfg
        )

        results.append({
            'name': name,
            'knn_wo': acc["knn_wo"] * 100,
            'svm_wo': acc["svm_wo"] * 100,
            'knn_w': acc["knn_w"] * 100,
            'svm_w': acc["svm_w"] * 100,
            'combined': acc["combined"] * 100,
            'note': note
        })

    # ========= 打印汇总表 =========
    print("\n\n================ Summary =================")
    print(
        f"{'Dataset':35s} | {'KNN wo/v':>8s} | {'SVM wo/v':>8s} | {'KNN w/v':>8s} | {'SVM w/v':>8s} | {'Combined':>8s} | Notes")
    print("-" * 100)
    for result in results:
        print(f"{result['name']:35s} | {result['knn_wo']:8.2f} | {result['svm_wo']:8.2f} | "
              f"{result['knn_w']:8.2f} | {result['svm_w']:8.2f} | {result['combined']:8.2f} | {result['note']}")
    print("=" * 100)




if __name__ == "__main__":
    run_multi_dataset_eval()
