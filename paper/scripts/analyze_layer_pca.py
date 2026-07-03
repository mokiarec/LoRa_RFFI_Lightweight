# paper/analyze_layer_pca.py
import os
import sys
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from paths import PathManager

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import PreprocessType, DEVICE
from net import NetworkType
from utils.data_preprocessor import load_generate_triplet, load_model
from dataset import *

path_manager = PathManager()

def extract_linear_and_conv_layers(model):
    """提取模型中所有线性层和卷积层"""
    layers = []
    for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         layers.append((name, module, 'Linear'))
        if isinstance(module, torch.nn.Conv2d):
            layers.append((name, module, 'Conv2d'))
    return layers

def analyze_layer_pca(layer_name, layer, input_data, layer_type, max_components=64):
    """分析单个层的PCA结果"""
    # 获取该层的输出特征
    with torch.no_grad():
        output = layer(input_data)
        
        # 对于Conv2d层，使用全局平均池化
        if layer_type == 'Conv2d':
            # [B, C, H, W] -> [B, C]
            features = output.mean(dim=[2, 3]).cpu().numpy()
        else:
            features = output.cpu().numpy()
    
    # 执行PCA
    n_components = min(max_components, features.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    # 计算累计方差贡献率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # 找到有效维度（累计方差达到95%的维度数）
    effective_dims_95 = np.argmax(cumulative_variance >= 0.95) + 1
    effective_dims_99 = np.argmax(cumulative_variance >= 0.99) + 1
    
    return {
        'layer_name': layer_name,
        'layer_type': layer_type,
        'input_dim': input_data.shape[1],
        'output_dim': features.shape[1],
        'effective_dims_95': effective_dims_95,
        'effective_dims_99': effective_dims_99,
        'cumulative_variance': cumulative_variance,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }

def plot_layer_pca_waveform(results, exp_name=""):
    """绘制各层PCA有效维度的波形图"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # 提取数据
    layer_names = [r['layer_name'] for r in results]
    effective_dims_95 = [r['effective_dims_95'] for r in results]
    effective_dims_99 = [r['effective_dims_99'] for r in results]
    output_dims = [r['output_dim'] for r in results]
    
    x = np.arange(len(layer_names))
    
    # 绘制波形图
    ax.plot(x, output_dims, 'o-', linewidth=1.5, markersize=6, label='Output Dim', color='#1f77b4')
    ax.plot(x, effective_dims_99, 's-', linewidth=1.5, markersize=6, label='99% Variance', color='#ff7f0e')
    ax.plot(x, effective_dims_95, '^-', linewidth=1.5, markersize=6, label='95% Variance', color='#2ca02c')
    
    ax.set_xlabel('Layer Name', fontsize=11)
    ax.set_ylabel('Dimensions', fontsize=11)
    title = f'PCA Effective Dimensions Across Layers\n{exp_name}' if exp_name else 'PCA Effective Dimensions Across Layers'
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=60, ha='right', fontsize=7)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=9, direction='in')
    
    plt.tight_layout()
    
    output_path = path_manager.get_paper_output_files().get('pca_layers', '.')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.show()

def main():
    # 数据集配置
    file_path_enrol = str(DATASET['Test']['seen'].path)
    dev_range_enrol = np.arange(0, 40)
    pkt_range_enrol = np.arange(0, 10)

    
    net_type = NetworkType.ResNet
    # model_path = "D:\\ScienceProject\\LoRa_RFFI\\checkpoints\\EXP_26_GoogleNet_Base\\weights\\Extractor_best.pth"
    model_path = "best.pth"
    preprocess_type = PreprocessType.STFT
    
    # 从模型路径提取实验名称
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    print(f"\nExperiment: {exp_name}")
    
    # 加载数据
    print("\nLoading data...")
    label_enrol, triplet_data_enrol = load_generate_triplet(
        file_path_enrol, dev_range_enrol, pkt_range_enrol,
        preprocess_type, snr_range=None
    )
    
    # 加载模型
    print("Loading model...")
    model = load_model(str(model_path), net_type, preprocess_type)
    model = model.to(DEVICE)
    triplet_data_enrol = [data.to(DEVICE) for data in triplet_data_enrol]
    
    # 提取embedding网络
    embedding_net = model.embedding_net
    embedding_net.eval()
    
    # 获取anchor数据
    anchor_data = triplet_data_enrol[0]
    
    # 提取所有线性层和卷积层
    all_layers = extract_linear_and_conv_layers(embedding_net)
    print(f"\nFound {len(all_layers)} layers:")
    for name, layer, layer_type in all_layers:
        if layer_type == 'Linear':
            print(f"  - {name} [{layer_type}]: {layer.in_features} -> {layer.out_features}")
        else:
            print(f"  - {name} [{layer_type}]: {layer.in_channels} -> {layer.out_channels}, kernel={layer.kernel_size}")
    
    # 逐层分析PCA
    print("\nAnalyzing PCA for each layer...")
    results = []
    
    # 需要通过forward hook来获取每层的输入
    hooks = []
    layer_inputs = {}
    
    def create_hook(name):
        def hook(module, input, output):
            layer_inputs[name] = input[0].detach()
        return hook
    
    # 注册hook
    for name, layer, layer_type in all_layers:
        hook = layer.register_forward_hook(create_hook(name))
        hooks.append(hook)
    
    # 前向传播
    with torch.no_grad():
        _ = embedding_net(anchor_data)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()

    # 分析每层
    for name, layer, layer_type in all_layers:
        if name in layer_inputs:
            input_data = layer_inputs[name]
            result = analyze_layer_pca(name, layer, input_data, layer_type)
            results.append(result)
    
    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY: Effective Dimensions Analysis")
    print("="*60)
    for r in results:
        print(f"{r['layer_name']:30s} | {r['layer_type']:7s} | Out: {r['output_dim']:4d} | "
              f"95%: {r['effective_dims_95']:4d} | 99%: {r['effective_dims_99']:4d}")
    print("="*60)
    
    # 绘图
    if results:
        plot_layer_pca_waveform(results, exp_name)

if __name__ == '__main__':
    main()
