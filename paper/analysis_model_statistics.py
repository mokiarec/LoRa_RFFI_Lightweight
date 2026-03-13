# paper/analysis_model_statistics.py
import os
import sys

import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import NetworkType, PreprocessType
from net.TripletNet import TripletNet
from training_utils.data_preprocessor import load_generate_triplet
from utils.FLOPs import calculate_flops_and_params
from paths import DATASET_FILES, PAPER_OUTPUT_FILES

def collect_model_statistics(model_dir, net_type, preprocess_type, file_path_enrol,
                           dev_range_enrol, pkt_range_enrol, width_multiplier=None):
    """
    收集单个模型的统计信息，包括参数量、存储大小和FLOPs
    """
    stats = {}

    # 加载数据用于FLOPs计算
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type
    )

    model_path = os.path.join(model_dir, f"{net_type.value}.pth")

    """创建模型但不加载权重"""
    if width_multiplier is not None:
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels, width_multiplier=width_multiplier)
    else:
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
    torch.save(model.state_dict(), model_path)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型文件大小（KB）
    file_size_mb = os.path.getsize(model_path) / 1024

    # 计算FLOPs（使用已有函数）
    flops, params_count = calculate_flops_and_params(model, triplet_data_enrol)

    stats[net_type.value] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_mb': file_size_mb,
        'flops': flops,
        'params_from_flops': params_count
    }

    print(f"统计完成: {net_type.value}")

    return stats

def collect_all_models_statistics():
    """
    收集所有模型的统计信息
    """

    # 定义所有需要统计的模型配置
    model_configs = [
        NetworkType.RESNET,
        NetworkType.MobileNetV1,
        NetworkType.MobileNetV2,
        NetworkType.MobileNetV3,
        NetworkType.MobileNetV4,
        NetworkType.LightNet,
        NetworkType.DRSN
    ]

    all_stats = {}

    for model_config in model_configs:
        print(f"\n正在收集 {model_config.value} 的统计信息...")

        stats = collect_model_statistics(
            model_dir='model',
            net_type=model_config,
            preprocess_type=PreprocessType.STFT,
            file_path_enrol=str(DATASET_FILES['train_no_aug']),
            dev_range_enrol=np.arange(0, 10, dtype=int),
            pkt_range_enrol=np.arange(0, 10, dtype=int),
        )

        all_stats[model_config.value] = stats

    return all_stats

def print_statistics_table(all_stats):
    """
    以表格形式打印统计信息
    """
    print("\n" + "="*120)
    print(f"{'模型名称':<25} {'参数量(K)':<12} {'文件大小(KB)':<15} {'FLOPs(K)':<12}")
    print("="*120)

    for model_name, stats in all_stats.items():
        if not stats:
            print(f"{model_name:<25} {'无模型文件':<8} {'-':<12} {'-':<15} {'-':<12}")
            continue

        for model_key, stat in stats.items():
            params_m = stat['total_params'] / 1e3
            flops_g = stat['flops'] / 1e3
            print(f"{model_name:<25} {params_m:<12.2f} {stat['file_size_mb']:<15.2f} {flops_g:<12.2f}")

    print("="*120)

def save_statistics_to_csv(all_stats, filename=None):
    """
    将统计信息保存到CSV文件
    """
    import csv

    if filename is None:
        filename = PAPER_OUTPUT_FILES['model_statistics']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'epoch', 'total_params', 'trainable_params',
                     'file_size_mb', 'flops', 'params_from_flops']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model_name, stats in all_stats.items():
            for model_key, stat in stats.items():
                row = {
                    'model_name': model_name,
                    'total_params': stat['total_params'],
                    'trainable_params': stat['trainable_params'],
                    'file_size_mb': stat['file_size_mb'],
                    'flops': stat['flops'],
                    'params_from_flops': stat['params_from_flops']
                }
                writer.writerow(row)

    print(f"\n统计信息已保存到 {filename}")

if __name__ == "__main__":
    # 收集所有模型的统计信息
    all_stats = collect_all_models_statistics()

    # 打印统计表格
    print_statistics_table(all_stats)

    # 保存到CSV文件
    save_statistics_to_csv(all_stats)

    # 打印汇总信息
    print("\n统计完成！")
