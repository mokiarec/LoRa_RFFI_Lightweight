"""项目主入口文件"""
import os
import numpy as np

from core.config import set_seed, Config, Mode, DEVICE, PreprocessType
from core.controller import main
from net import NetworkType


if __name__ == "__main__":
    # 设置随机种子确保可重现性
    set_seed(42)

    # 可以通过修改这里的参数来选择运行模式:
    # Mode.TRAIN, Mode.CLASSIFICATION, Mode.ROGUE_DEVICE_DETECTION, Mode.DISTILLATION

    # ==================== 配置选择开关 ====================
    # 修改此变量来选择不同的实验配置
    EXP_SELECT = 5
    EXP_resume = False
    EXP_id = None

    # ==================== 各实验配置定义 ====================
    if EXP_SELECT == 1:
        # --- 示例 1: 训练基础模型 (EXP_01) ---
        config = Config(
            mode=Mode.TRAIN,
            net_type=NetworkType.ShuffleNet,
            preprocess_type=PreprocessType.STFT,
            learning_rate=1e-4,
            batch_size=32,
            exp_description="Base",
            snr=np.arange(20, 80),
        )
        config.save_to_json()

    elif EXP_SELECT == 2:
        # --- 示例 2: 基于 EXP_01 做剪枝实验 (EXP_02) ---
        config = Config(
            mode=Mode.TRAIN,
            base_net_type=NetworkType.ResNet,
            exp_description="Pruning",
            base_version="01"  # 基于 EXP_01
        )

    elif EXP_SELECT == 3:
        # --- 示例 3: 知识蒸馏实验 (EXP_03) ---
        config = Config(
            mode=Mode.DISTILLATION,
            net_type=NetworkType.ShuffleNet_prune,
            preprocess_type=PreprocessType.STFT,
            exp_description="KD_PCA8",
            base_version="24",  # 基于 EXP_02
            # snr=np.arange(20, 80),
            is_pca_train=True,
        )
        config.save_to_json()

    elif EXP_SELECT == 4:
        # --- 示例 4: 分类实验 (EXP_01) ---
        config = Config.from_json(
            mode=Mode.CLASSIFICATION,
            model_dir="./checkpoints/EXP_01_LightNet_Base"
        )
        EXP_resume = True
        EXP_id = "pwqr0un0fkph9ce2zz9t6"

    elif EXP_SELECT == 5:
        # --- 示例 5: 多分类实验 (EXP_01) ---
        config = Config.from_json(
            mode=Mode.MULTI_CLASSIFICATION,
            model_dir="./checkpoints/EXP_24_ShuffleNet_Base"
        )
        EXP_resume = True
        EXP_id = "dr9l190g4h54u9xonjkqg"

    elif EXP_SELECT == 6:
        # --- 示例 6: 分类实验 (EXP_01) ---
        config = Config.from_json(
            mode=Mode.CLASSIFICATION,
            model_dir="./checkpoints/EXP_17_ResNet_prune_v02_KD_PCA8"
        )

    else:
        raise ValueError(f"未找到 EXPERIMENT_SELECT = {EXP_SELECT} 对应的配置")

    try:
        import swanlab
        SWANLAB_AVAILABLE = True
    except ImportError:
        SWANLAB_AVAILABLE = False
        print("警告：未安装 swanlab，将跳过实验跟踪")

    # 初始化 SwanLab
    if SWANLAB_AVAILABLE:
        # 动态生成：例如 EXP_01_LightNet_Base_classification
        mode_name = config.mode.value if isinstance(config.mode, Mode) else str(config.mode)
        current_exp_name = f"{config.EXP_NAME}"
        custom_logdir = os.path.join(config.MODEL_DIR, "swanlog")
        try:
            swanlab.login()
            swanlab.init(
                project="Lightweight_LoRa_RFFI",
                experiment_name=current_exp_name,
                resume=EXP_resume,
                id=EXP_id,
                config={
                    "network_type": config.NET_TYPE.value,
                    "preprocess_type": config.PREPROCESS_TYPE.value,
                    "batch_size": config.HP['batch_size'],
                    "num_epochs": config.HP['num_epochs'],
                    "learning_rate": config.HP['learning_rate'],
                    "temperature": config.HP['temperature'],
                    "alpha": config.HP['alpha'],
                    "snr": config.HP['snr'],
                    "pca_dim_train": config.PCA_DIM_TRAIN if config.IS_PCA_TRAIN else None,
                    "test_list": config.TEST_LIST,
                    "device": str(DEVICE),
                },
                logdir=custom_logdir
            )
        except Exception as e:
            print(f"SwanLab 初始化失败：{e}")
            SWANLAB_AVAILABLE = False

    # 运行选定的配置
    main(config)

    if SWANLAB_AVAILABLE:
        swanlab.finish()