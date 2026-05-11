from enum import Enum

# 定义运行模式的枚举
class Mode(str, Enum):
    """运行模式枚举"""
    TRAIN = "train"                    # 训练模式 - 用于训练基础模型
    CLASSIFICATION = "clf"  # 分类模式 - 用于设备指纹分类任务
    MULTI_CLASSIFICATION = "multi_clf"  # 多数据集分类评估模式 - 用于跨场景泛化能力测试
    ROGUE_DEVICE_DETECTION = "rogue"  # 恶意设备检测模式 - 用于检测非法设备
    DISTILLATION = "distill"      # 蒸馏模式 - 用于知识蒸馏训练轻量级模型
    LATENCY_BENCHMARK = "benchmark"   # 延迟基准测试模式 - 用于评估模型推理速度


# 定义预处理类型枚举
class PreprocessType(Enum):
    """预处理类型枚举"""
    IQ = ("IQ", 1)      # IQ数据直接使用，2个通道（I和Q）
    STFT = ("STFT", 1)  # 短时傅里叶变换，1个通道（幅度）
    WST = ("WST", 2)    # 小波散射变换，2个通道（实部和虚部）

    def __init__(self, name, in_channels):
        self._value_ = name
        self.in_channels = in_channels