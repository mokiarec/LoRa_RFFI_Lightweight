# paper/analysis_benchmark_latency.py
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config, Mode
from modes.classification_mode import test_classification
from paths import DATASET_FILES, get_model_path

def run_latency_benchmark(num_runs=10, **kwargs):
    """
    多次运行分类测试以统计平均延迟。

    :param num_runs: 运行次数
    :param kwargs: 传递给 test_classification 的参数
    """

    enrol_times = []
    predict_times = []

    print(f"Starting benchmark with {num_runs} runs...")

    # 预热 (Warm-up) - 可选
    # 有时候第一次运行会因为加载库或缓存原因较慢，通常先跑一次不计入统计
    print("Warming up...")
    test_classification(enable_plots=False, **kwargs)  # 关闭绘图以节省时间

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}...")

        # 强制关闭绘图，避免产生大量图片文件拖慢速度
        kwargs['enable_plots'] = False

        # 运行测试
        results = test_classification(**kwargs)

        # 收集时间
        enrol_times.append(results['time_enrol'])
        predict_times.append(results['time_predict'] + results['time_clf_feat'])

    # 计算统计数据
    avg_enrol = np.mean(enrol_times)
    std_enrol = np.std(enrol_times)

    avg_predict = np.mean(predict_times)
    std_predict = np.std(predict_times)

    print("\n" + "=" * 40)
    print("BENCHMARK RESULTS")
    print("=" * 40)
    print(f"Total Runs: {num_runs}")
    print("-" * 20)
    print(f"Enrollment Time (Total):")
    print(f"  Mean: {avg_enrol:.4f} s")
    print(f"  Std : {std_enrol:.4f} s")
    print("-" * 20)
    print(f"Prediction Time (Total):")
    print(f"  Mean: {avg_predict:.4f} s")
    print(f"  Std : {std_predict:.4f} s")
    print("=" * 40)


# --- 使用示例 ---
if __name__ == "__main__":
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.DISTILLATION
    config = Config(Mode.CLASSIFICATION)

    # 配置测试参数
    benchmark_params = {
        "mode": config.mode,
        "file_path_enrol": str(DATASET_FILES['train_aug']),
        "file_path_clf": str(DATASET_FILES['test_seen']),
        "dev_range_enrol": np.arange(0, 40, dtype=int),
        "pkt_range_enrol": np.arange(0, 200, dtype=int),
        "dev_range_clf": np.arange(0, 40, dtype=int),
        "pkt_range_clf": np.arange(0, 100, dtype=int),
        "net_type": config.NET_TYPE,
        "preprocess_type": config.PREPROCESS_TYPE,
        "test_list": config.TEST_LIST,
        "model_dir": get_model_path(config.PREPROCESS_TYPE, config.NET_TYPE),
        "is_pac": config.IS_PCA_TEST,
    }

    run_latency_benchmark(num_runs=10, **benchmark_params)
