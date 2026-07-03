# 工具包初始化文件

# 导出 SwanLab 相关模块
from .swanlab_manager import swanlab_manager, SwanLabManager
from .training_decorators import (
    auto_log_training,
    track_experiment,
    combined_training_decorator
)

__all__ = [
    'swanlab_manager',
    'SwanLabManager',
    'auto_log_training',
    'track_experiment',
    'combined_training_decorator',
]