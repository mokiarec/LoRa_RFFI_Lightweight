# paper/analysis_model_statistics.py
import os
import sys

import torch

from paths import PathManager

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net import TripletNet
from utils.data_preprocessor import load_generate_triplet
from utils.FLOPs import calculate_flops_and_params
from dataset import *


def collect_model_statistics_from_checkpoint(checkpoint_dir, net_type, preprocess_type, file_path_enrol,
                                           dev_range_enrol, pkt_range_enrol, width_multiplier=None):
    """
    从checkpoint目录收集单个模型的统计信息，包括参数量、存储大小和FLOPs
    """
    stats = {}

    # 加载数据用于FLOPs计算
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type
    )

    # 查找权重目录中的模型文件
    weights_dir = os.path.join(checkpoint_dir, 'weights')
    if not os.path.exists(weights_dir):
        print(f"权重目录不存在: {weights_dir}")
        return stats
    
    # 查找最佳模型或最后一个epoch的模型
    model_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth') and not f.startswith('pca_')]
    if not model_files:
        print(f"在 {weights_dir} 中未找到模型文件")
        return stats
    
    # 优先选择best模型，否则选择最新的epoch模型
    best_model = None
    latest_model = None
    max_epoch = -1
    
    for model_file in model_files:
        if 'best' in model_file.lower():
            best_model = model_file
        else:
            # 尝试提取epoch数字
            try:
                epoch_str = model_file.replace('Extractor_', '').replace('.pth', '')
                epoch_num = int(epoch_str)
                if epoch_num > max_epoch:
                    max_epoch = epoch_num
                    latest_model = model_file
            except ValueError:
                continue
    
    # 选择模型文件
    selected_model = best_model if best_model else latest_model
    if not selected_model:
        selected_model = model_files[0]  # 如果都没匹配到，选第一个
    
    model_path = os.path.join(weights_dir, selected_model)
    print(f"使用模型文件: {selected_model}")

    """创建模型并加载权重"""
    if width_multiplier is not None:
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels, width_multiplier=width_multiplier)
    else:
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        return stats

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算模型文件大小（KB）
    file_size_kb = os.path.getsize(model_path) / 1024

    # 计算FLOPs（使用已有函数）
    flops, params_count = calculate_flops_and_params(model, triplet_data_enrol)

    stats[selected_model] = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'file_size_kb': file_size_kb,
        'flops': flops,
        'params_from_flops': params_count
    }

    print(f"统计完成: {net_type.value} - {selected_model}")

    return stats

def collect_specific_checkpoints_statistics(exp_numbers=None):
    """
    收集指定序号的EXP实验目录下checkpoint模型的统计信息
    参数 exp_numbers: 实验序号列表，例如 [1, 5, 10] 表示分析 EXP_01, EXP_05, EXP_10
    如果为None，则收集所有实验
    """
    import json
    
    all_stats = {}
    checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
    
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints目录不存在: {checkpoints_dir}")
        return all_stats
    
    # 根据指定的实验序号生成目录名列表
    if exp_numbers is not None:
        exp_dirs = []
        for num in exp_numbers:
            exp_dir_name = f"EXP_{num:02d}_"
            # 在checkpoints目录中查找以该前缀开头的目录
            matching_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith(exp_dir_name)]
            exp_dirs.extend(matching_dirs)
        exp_dirs = list(set(exp_dirs))  # 去重
    else:
        # 遍历所有EXP实验目录
        exp_dirs = [d for d in os.listdir(checkpoints_dir) if d.startswith('EXP_')]
    
    exp_dirs.sort()  # 按名称排序
    
    for exp_dir_name in exp_dirs:
        exp_dir_path = os.path.join(checkpoints_dir, exp_dir_name)
        if not os.path.isdir(exp_dir_path):
            continue
            
        print(f"\n正在处理实验目录: {exp_dir_name}")
        
        # 读取config.json获取网络类型等信息
        config_path = os.path.join(exp_dir_path, 'config.json')
        if not os.path.exists(config_path):
            print(f"  配置文件不存在: {config_path}，跳过")
            continue
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"  读取配置文件失败: {e}，跳过")
            continue
        
        # 获取网络类型
        net_type_str = config.get('NET_TYPE', '')
        if not net_type_str:
            print(f"  配置中缺少NET_TYPE，跳过")
            continue
        
        # 将字符串转换为NetworkType枚举
        try:
            from net import NetworkType
            net_type = NetworkType(net_type_str)
        except ValueError:
            print(f"  未知的网络类型: {net_type_str}，跳过")
            continue
        
        # 获取预处理类型
        preprocess_type_str = config.get('PREPROCESS_TYPE', 'STFT')
        from core.config import PreprocessType
        try:
            preprocess_type = PreprocessType(preprocess_type_str)
        except ValueError:
            print(f"  未知的预处理类型: {preprocess_type_str}，使用默认STFT")
            preprocess_type = PreprocessType.STFT
        
        # 收集该实验的统计信息
        stats = collect_model_statistics_from_checkpoint(
            checkpoint_dir=exp_dir_path,
            net_type=net_type,
            preprocess_type=preprocess_type,
            file_path_enrol=str(DATASET['Train']['no_aug'].path),
            dev_range_enrol=np.arange(0, 10, dtype=int),
            pkt_range_enrol=np.arange(0, 10, dtype=int),
        )
        
        if stats:
            all_stats[exp_dir_name] = {
                'net_type': net_type_str,
                'preprocess_type': preprocess_type_str,
                'models': stats
            }

    return all_stats

def print_statistics_table(all_stats):
    """
    以表格形式打印统计信息
    """
    print("\n" + "="*140)
    print(f"{'实验名称':<35} {'模型文件':<30} {'参数量(K)':<12} {'文件大小(KB)':<15} {'FLOPs(K)':<12}")
    print("="*140)

    for exp_name, exp_data in all_stats.items():
        net_type = exp_data['net_type']
        models = exp_data['models']
        
        if not models:
            print(f"{exp_name:<35} {'无模型文件':<30} {'-':<12} {'-':<15} {'-':<12}")
            continue

        for model_key, stat in models.items():
            params_k = stat['total_params'] / 1e3
            flops_k = stat['flops'] / 1e3
            print(f"{exp_name:<35} {model_key:<30} {params_k:<12.2f} {stat['file_size_kb']:<15.2f} {flops_k:<12.2f}")

    print("="*140)

def save_statistics_to_csv(all_stats, filename=None):
    """
    将统计信息保存到CSV文件
    """
    import csv

    if filename is None:
        filename = PathManager.get_paper_output_files()['model_statistics']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['exp_name', 'net_type', 'preprocess_type', 'model_file', 'total_params', 'trainable_params',
                     'file_size_kb', 'flops', 'params_from_flops']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for exp_name, exp_data in all_stats.items():
            net_type = exp_data['net_type']
            preprocess_type = exp_data['preprocess_type']
            models = exp_data['models']
            
            for model_key, stat in models.items():
                row = {
                    'exp_name': exp_name,
                    'net_type': net_type,
                    'preprocess_type': preprocess_type,
                    'model_file': model_key,
                    'total_params': stat['total_params'],
                    'trainable_params': stat['trainable_params'],
                    'file_size_kb': stat['file_size_kb'],
                    'flops': stat['flops'],
                    'params_from_flops': stat['params_from_flops']
                }
                writer.writerow(row)

    print(f"\n统计信息已保存到 {filename}")

if __name__ == "__main__":
    # 指定要分析的EXP实验序号，例如：[1, 5, 10]
    # 设置为 None 则分析所有实验
    EXP_NUMBERS = [2, 17, 20, 21, 22, 23, 24, 25]  # 修改这里来指定要分析的实验序号
    
    print(f"开始分析实验: {EXP_NUMBERS if EXP_NUMBERS else '所有实验'}")
    
    # 收集指定EXP实验目录下checkpoint模型的统计信息
    all_stats = collect_specific_checkpoints_statistics(exp_numbers=EXP_NUMBERS)

    # 打印统计表格
    print_statistics_table(all_stats)

    # 保存到CSV文件
    save_statistics_to_csv(all_stats)

    # 打印汇总信息
    print(f"\n共处理 {len(all_stats)} 个实验目录")
