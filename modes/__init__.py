# 模式模块初始化文件
from modes.classification_mode import test_classification
from modes.distillation_mode import distillation, finetune_with_awgn
from modes.latency_benchmark_mode import test_latency_benchmark
from modes.multi_clf_mode import test_multi_clf
from modes.rogue_device_detection_mode import test_rogue_device_detection
from modes.train_mode import train

__all__ = [
    'train',
    'test_classification',
    'test_multi_clf',
    'test_rogue_device_detection',
    'distillation',
    'finetune_with_awgn',
    'test_latency_benchmark',
]
