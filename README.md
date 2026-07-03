# LoRa_RFFI 轻量化项目

基于 **PCA驱动的通道剪枝 + 自蒸馏** 的自动化轻量级射频指纹识别（RFFI）系统。该流水线自动从高容量教师网络中提取紧凑学生架构，无需人工设计轻量网络或启发式宽度乘数。

## 流水线概述

详见论文 `paper/main.tex`，核心流程如下：

1. **训练高容量教师网络** — 使用三元组损失训练大网络，建立高判别性射频指纹嵌入空间。
2. **逐层 PCA 通道诊断** — 对每个 Conv2d 层的输出特征图执行主成分分析，以累积方差阈值（默认 95%）揭示每层所需的有效通道维度。
3. **结构剪枝 + 随机初始化** — 根据 PCA 推导的通道配置构建紧凑学生网络，权重完全随机初始化（无参数继承）。
4. **PCA 对齐自蒸馏** — 通过 PCA 将教师高维嵌入投影到学生低维空间，用温度缩放 KL 散度蒸馏，使随机初始化的学生恢复教师级判别力。
5. **迭代压缩（可选）** — 链式执行多轮"诊断→剪枝→蒸馏"，在可配置准确率阈值内实现渐进式压缩（如 ResNet 三轮从 203K → 1.4K 参数）。

支持架构：ResNet, SCSKNet, DenseNet, ShuffleNetV2, GoogleNet, MobileNetV1/V2, LightNet。

## 目录结构

```
LoRa_RFFI/
├── core/                           # 核心配置和控制器
│   ├── config.py                   # 全局配置、超参数管理
│   └── controller.py               # 主控制器，调度各模式执行
├── modes/                          # 运行模式实现
│   ├── train_mode.py               # 基础训练（三元组损失 + KNN 验证）
│   ├── classification_mode.py      # 单场景分类评估
│   ├── multi_clf_mode.py           # 多场景/跨场景分类评估
│   ├── distillation_mode.py        # 旧版 PCA 蒸馏（checkpoints 方式）
│   ├── rogue_device_detection_mode.py # 恶意设备检测
│   └── latency_benchmark_mode.py   # Jetson/GPU 延迟基准测试
├── net/                            # 神经网络模型定义
│   ├── net_ResNet.py               # ResNet 系列
│   ├── net_SCSKNet.py              # SCSKNet 系列
│   ├── net_DenseNet.py             # DenseNet 系列
│   ├── net_ShuffleNet.py           # ShuffleNetV2 系列
│   ├── net_GoogleNet.py            # GoogleNet 系列
│   ├── net_MobileNet.py            # MobileNetV1/V2、LightNet 系列
│   ├── net_DRSN.py                 # DRSN 系列
│   ├── net_VGG.py                  # VGG 系列
│   └── net_wt.py                   # 小波变换网络
├── pipeline/                       # ✨ 核心轻量化流水线
│   ├── main.py                     # 流水线主入口（单次/迭代模式，CLI 支持）
│   ├── pca_conv.py                 # 逐层 Conv2d PCA 诊断
│   └── prune_builder.py            # 动态通道剪枝网络构建器
├── dataset/                        # 数据集加载与预处理
├── utils/                          # 通用工具函数
│   ├── data_preprocessor.py        # 数据预处理（IQ/STFT/WST）
│   ├── data_loader.py              # 数据加载器
│   ├── TripletDataset.py           # 三元组数据集与损失
│   ├── PCA.py                      # 嵌入空间 PCA 工具
│   ├── FLOPs.py                    # FLOPs 与参数量计算
│   ├── TSNE.py                     # t-SNE 可视化
│   ├── signal_trans.py             # 信号变换
│   ├── model_utils.py              # 模型加载/保存工具
│   └── swanlab_manager.py          # SwanLab 实验跟踪
├── plot/                           # 可视化工具
│   ├── plot_confusion.py           # 混淆矩阵
│   ├── plot_loss.py                # 损失曲线
│   └── plot_roc.py                 # ROC 曲线
├── checkpoints/                    # 实验检查点（权重 + 配置 + 评估结果）
├── paper/                          # 论文 LaTeX 源码与图表
├── jetson/                         # Jetson 边缘部署测试
│   └── jetson_benchmark.py         # Jetson 推理基准测试
├── main.py                         # (旧版入口) 通过 config 选择训练/分类/蒸馏等模式
├── paths.py                        # 统一路径管理
└── requirements.txt                # Python 依赖
```

## 使用方法

### 方式一（推荐）：轻量化流水线

使用 `pipeline/main.py` 的自动流水线，支持 CLI 参数：

#### 单次蒸馏
```bash
# 从 ResNet 教师网络 → PCA 诊断 → 剪枝构建学生 → 自蒸馏 → 评估
python pipeline/main.py \
    --mode single \
    --net ResNet \
    --epochs 300 \
    --batch-size 32 \
    --pca-threshold 0.95 \
    --alpha 0.7 \
    --temperature 3.0 \
    --embedding-dim 8 \
    --min-channels 2
```

#### 迭代蒸馏
```bash
# 链式多轮压缩，直到准确率下降超过阈值
python pipeline/main.py \
    --mode iterative \
    --net ResNet \
    --epochs 100 \
    --batch-size 32 \
    --pca-threshold 0.95 \
    --accuracy-threshold 5.0 \
    --max-iterations 10
```

#### 从检查点恢复迭代
```bash
python pipeline/main.py \
    --mode iterative \
    --net ResNet \
    --resume-from ./pipeline/runs_iterative/run_001_ResNet_iterative
```

### 方式二：旧版实验管理（checkpoints 方式）

通过 `main.py` 中的 `EXP_SELECT` 选择预定义实验配置：

```python
# 在 main.py 中设置 EXP_SELECT
# 1 = 训练基础模型 (TRAIN)
# 3 = 知识蒸馏 (DISTILLATION)
# 4 = 单场景分类评估 (CLASSIFICATION)
# 5 = 多场景分类评估 (MULTI_CLASSIFICATION)
# 6 = 分类评估指定实验

EXP_SELECT = 5
```

然后运行：
```bash
python main.py
```

### 方式三：Jetson 边缘部署测试
```bash
python jetson/jetson_benchmark.py --model-path <模型路径>
```

## 核心参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pca-threshold` | 0.95 | PCA 累积方差阈值，决定每层有效通道数 |
| `embedding-dim` | 8 | 学生网络输出嵌入维度 |
| `alpha` | 0.7 | 蒸馏损失权重（0=仅三元组，1=仅 KL） |
| `temperature` | 3.0 | KL 蒸馏温度参数 |
| `min-channels` | 2 | 每层最小通道数下限 |
| `accuracy-threshold` | 5.0% | 迭代模式中可接受的准确率下降阈值 |

## 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA (推荐) + cuDNN
- 详见 `requirements.txt`
