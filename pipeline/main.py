#!/usr/bin/env python3
"""
流水线：大网络探索 → PCA诊断冗余 → 结构剪枝 → 随机初始化 → 自蒸馏 → 评估

步骤:
  1. 训练大网络（三元组损失 + embedding + KNN 验证）
  2. 对每个 Conv2d 输出做 PCA —— 找到有效通道维度
  3. 根据 PCA 有效维度构建剪枝后的小网络（随机初始化权重）
  4. 自蒸馏：大网络教小网络（KL 散度 + 三元组损失）
  5. 评估两个网络，对比准确率与参数量
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从父项目导入（不修改原有代码）
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import set_seed, PreprocessType, DEVICE
from dataset import DATASET
from modes.train_mode import train as train_large
from net import NetworkType, TripletNet
from net.net_ResNet import ResNet, ResNet_prune
from net.net_SCSKNet import SCSKNet, SCSKNet_prune
from net.net_GoogleNet import GoogleNet, GoogleNet_prune
from net.net_DenseNet import DenseNet, DenseNet_prune
from net.net_ShuffleNet import ShuffleNetV2, ShuffleNetV2_prune
from net.net_MobileNet import MobileNetV1, MobileNetV2, LightNet, LightNet_prune
from utils.data_preprocessor import prepare_train_data
from utils.TripletDataset import TripletDataset, TripletLoss

from pipeline.pca_conv import analyze_conv_redundancy, print_analysis
from pipeline.prune_builder import (
    build_pruned_embedding_net,
    derive_channels_from_pca,
)

# ────────────────────────────────────────────────────────────
# 原始网络 → 已有剪枝网络（硬编码版本，作为 fallback/对比）
# ────────────────────────────────────────────────────────────

NET_PAIRS = {
    'ResNet': (ResNet, ResNet_prune),
    'SCSKNet': (SCSKNet, SCSKNet_prune),
    'GoogleNet': (GoogleNet, GoogleNet_prune),
    'DenseNet': (DenseNet, DenseNet_prune),
    'ShuffleNet': (ShuffleNetV2, ShuffleNetV2_prune),
    'MobileNetV1': (MobileNetV1, LightNet_prune),
    'MobileNetV2': (MobileNetV2, LightNet_prune),
    'LightNet': (LightNet, LightNet_prune),
}


def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ────────────────────────────────────────────────────────────
# 辅助函数：智能加载模型
# ────────────────────────────────────────────────────────────

def _load_model_smart(model_path, net_type, preprocess_type, hp=None):
    """智能加载模型，支持原始格式和剪枝后的字典格式。

    Args:
        model_path: 模型文件路径
        net_type: 网络类型
        preprocess_type: 预处理类型
        hp: 超参数字典（用于剪枝模型的宽度乘数等）

    Returns:
        tuple: (model, is_pruned) - 模型对象和是否为剪枝模型的标志
    """
    saved = torch.load(model_path, weights_only=True)

    # 检查是否为剪枝模型的字典格式
    if isinstance(saved, dict) and 'state_dict' in saved:
        # 剪枝模型格式
        is_pruned = True
        channels = saved.get('channels', None)
        embedding_dim = saved.get('embedding_dim', 8)
        width_mult = saved.get('width_multiplier', 1 / 16)
        net_type_str = saved.get('net_type', net_type.value if hasattr(net_type, 'value') else str(net_type))

        # 构建剪枝网络
        if channels is not None:
            emb_net = build_pruned_embedding_net(
                net_type_str, preprocess_type.in_channels, channels, embedding_dim)
        else:
            # 回退到预定义剪枝架构
            _, prune_cls = NET_PAIRS[net_type_str]
            if net_type_str in ('MobileNetV1', 'MobileNetV2', 'LightNet'):
                emb_net = prune_cls(preprocess_type.in_channels, width_multiplier=width_mult)
            else:
                emb_net = prune_cls(preprocess_type.in_channels)

        emb_net.load_state_dict(saved['state_dict'])
        # 包装成 TripletNet
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
        model.embedding_net = emb_net
    else:
        # 原始模型格式
        is_pruned = False
        model = TripletNet(net_type=net_type, in_channels=preprocess_type.in_channels)
        model.load_state_dict(saved)

    model.to(DEVICE)
    model.eval()
    return model, is_pruned


# ────────────────────────────────────────────────────────────
# 步骤 1: 训练大网络
# ────────────────────────────────────────────────────────────

def step1_train_large(run_dir, net_type, data, labels, preprocess_type, hp):
    """用三元组损失训练大网络（教师）。

    复用现有的 train() 函数（modes/train_mode.py）。
    """
    print("\n" + "=" * 60)
    print("  步骤 1: 训练大网络")
    print("=" * 60)

    # 构造 train() 所需的最小配置对象
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.net_type = net_type
    cfg.preprocess_type = preprocess_type
    cfg.MODEL_WEIGHTS_DIR = os.path.join(run_dir, "large", "weights")
    cfg.MODEL_DIR = os.path.join(run_dir, "large")
    cfg.test_list = [1, 5, 10, 20, 35, 50, 60, 70, 85, 100, 150, 200]
    cfg.disable_swanlab = True

    os.makedirs(cfg.MODEL_WEIGHTS_DIR, exist_ok=True)

    train_large(cfg, data, labels,
                batch_size=hp['batch_size'],
                num_epochs=hp['num_epochs'],
                learning_rate=hp['learning_rate'])

    model_path = os.path.join(cfg.MODEL_WEIGHTS_DIR, "Extractor_best.pth")
    print(f"大网络模型已保存: {model_path}")
    return model_path


# ────────────────────────────────────────────────────────────
# 步骤 2: 逐层 Conv PCA 分析
# ────────────────────────────────────────────────────────────

def step2_pca_analysis(large_model_path, net_type, preprocess_type, data, labels, hp):
    """Hook 每个 Conv2d，收集输出，运行 PCA，报告有效维度。"""
    print("\n" + "=" * 60)
    print("  步骤 2: PCA 卷积层冗余分析")
    print("=" * 60)

    # 加载大网络（智能加载，支持原始和剪枝格式）
    model, is_pruned = _load_model_smart(large_model_path, net_type, preprocess_type, hp)
    
    if is_pruned:
        print(f"  注意：加载的是剪枝模型作为教师")
    
    model.to(DEVICE)
    model.eval()

    # 构造 dataloader
    dataset = TripletDataset(data, labels, PreprocessType.STFT)
    loader = DataLoader(dataset, batch_size=hp['batch_size'], shuffle=False)

    # 逐层 PCA
    results = analyze_conv_redundancy(
        model, loader, device=DEVICE,
        threshold=hp.get('pca_threshold', 0.95),
        max_spatial_samples=10000,
        max_batches=hp.get('pca_max_batches', 50),
    )

    print_analysis(results)
    return results


# ────────────────────────────────────────────────────────────
# 步骤 3+4: 构建剪枝网络 + 自蒸馏
# ────────────────────────────────────────────────────────────

def step34_distill(run_dir, large_model_path, pca_results,
                   net_type, preprocess_type, data, labels, hp):
    """用 PCA 结果构建剪枝学生网络，随机初始化后从教师蒸馏。

    学生网络由 build_pruned_embedding_net 根据 PCA 有效维度构建，
    权重为 PyTorch 默认随机初始化。

    蒸馏前先用 PCA 将教师高维 embedding（如 512d）压缩到学生维度（如 8d），
    使 KL 散度可在同一空间计算。
    训练损失:
        L = (1-α) * 三元组损失 + α * 蒸馏损失
    """
    print("\n" + "=" * 60)
    print("  步骤 3+4: 构建剪枝网络 & 自蒸馏")
    print("=" * 60)

    # ── 从 PCA 推导通道配置 ──
    net_type_str = net_type.value if hasattr(net_type, 'value') else str(net_type)
    try:
        channels = derive_channels_from_pca(pca_results, net_type=net_type_str,
                                             min_channels=hp.get('min_channels', 2))
        print(f"PCA 推导的通道配置: {channels}")
    except Exception as e:
        print(f"PCA 通道推导失败 ({e})，使用预定义剪枝架构")
        channels = None

    embedding_dim = hp.get('embedding_dim', 8)
    alpha = hp.get('alpha', 0.7)
    temperature = hp.get('temperature', 3.0)
    batch_size = hp['batch_size']
    num_epochs = hp['num_epochs']
    learning_rate = hp['learning_rate']
    width_mult = hp.get('width_multiplier', 1 / 16)  # MobileNet/LightNet 宽度乘数

    # ── 构建教师（冻结） ──
    teacher, is_pruned_teacher = _load_model_smart(large_model_path, net_type, preprocess_type, hp)
    
    if is_pruned_teacher:
        print(f"  注意：教师网络是剪枝模型")
    
    teacher.to(DEVICE)
    teacher.eval()

    # ── PCA 压缩教师 embedding：512d → embedding_dim ──
    print("计算教师 embedding 的 PCA 投影矩阵...")
    teach_dataset = TripletDataset(data, labels, PreprocessType.STFT)
    teach_loader = DataLoader(teach_dataset, batch_size=batch_size, shuffle=False)

    teach_feats = []
    with torch.no_grad():
        for anchor, _, _ in teach_loader:
            anchor = anchor.to(DEVICE)
            emb = teacher.embedding_net(anchor)
            teach_feats.append(emb.cpu().numpy())

    teach_feats = np.concatenate(teach_feats, axis=0)
    emb_pca = PCA(n_components=embedding_dim)
    emb_pca.fit(teach_feats)
    # W: (D_t, d), mean: (D_t,)
    W = torch.tensor(emb_pca.components_.T, dtype=torch.float32).to(DEVICE)
    mean = torch.tensor(emb_pca.mean_, dtype=torch.float32).to(DEVICE)
    print(f"PCA embedding 投影: {W.shape[0]} → {W.shape[1]}")

    # ── 构建学生（随机初始化） ──
    if channels is not None:
        student_emb = build_pruned_embedding_net(
            net_type_str, preprocess_type.in_channels, channels, embedding_dim)
        print(f"动态剪枝架构: {net_type_str}, 通道配置: {channels}")
    else:
        # 回退：使用已有的硬编码剪枝架构
        _, prune_cls = NET_PAIRS[net_type_str]
        if net_type_str in ('MobileNetV1', 'MobileNetV2', 'LightNet'):
            student_emb = prune_cls(preprocess_type.in_channels,
                                    width_multiplier=width_mult)
        else:
            student_emb = prune_cls(preprocess_type.in_channels)
        print(f"回退到预定义剪枝架构: {net_type_str}_prune")

    student_emb.to(DEVICE)
    n_params_student = sum(p.numel() for p in student_emb.parameters())
    print(f"学生网络参数量: {n_params_student:,}")

    # ── 数据准备 ──
    data_train, data_valid, labels_train, labels_valid = train_test_split(
        data, labels, test_size=0.1, shuffle=True)
    train_dataset = TripletDataset(data_train, labels_train, PreprocessType.STFT)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    batch_num = math.ceil(len(train_dataset) / batch_size)

    valid_data_tensor = torch.tensor(data_valid, dtype=torch.float32).to(DEVICE)
    valid_labels_tensor = torch.tensor(labels_valid, dtype=torch.long).to(DEVICE)

    # ── PCA 投影函数 ──
    def project_teacher(emb):
        """将教师高维 embedding 通过 PCA 投影到低维空间并 L2 归一化"""
        return F.normalize((emb - mean.unsqueeze(0)) @ W, p=2, dim=1)

    # ── 优化器与损失 ──
    optimizer = optim.Adam(student_emb.parameters(), lr=learning_rate)
    triplet_loss_fn = TripletLoss(margin=0.1)

    # ── 训练循环（带最小epoch和早停机制） ──
    min_epochs = hp.get('min_distill_epochs', 20)  # 最小蒸馏epoch数，确保小模型有足够时间学习
    patience = hp.get('early_stopping_patience', 10)  # 早停耐心值
    
    # 自动调整：如果min_epochs > num_epochs，则增加num_epochs
    if min_epochs > num_epochs:
        print(f"\n⚠️  警告: min_distill_epochs ({min_epochs}) > num_epochs ({num_epochs})")
        print(f"   自动调整 num_epochs 为 {min_epochs + patience} 以确保最小epoch要求")
        num_epochs = min_epochs + patience
    
    print(f"\n蒸馏参数: α={alpha}, T={temperature}")
    print(f"  最大epochs: {num_epochs}, 最小epochs: {min_epochs}, 早停patience: {patience}")
    
    loss_per_epoch = []
    best_accuracy = 0.0
    best_state_dict = None
    best_epoch = 0
    no_improve_count = 0  # 连续未改进的epoch计数
    weights_dir = os.path.join(run_dir, "pruned", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    with tqdm(total=num_epochs, desc="蒸馏进度") as total_bar:
        for epoch in range(num_epochs):
            student_emb.train()
            total_loss = 0.0

            for anchor, positive, negative in train_loader:
                anchor = anchor.to(DEVICE)
                positive = positive.to(DEVICE)
                negative = negative.to(DEVICE)

                # 教师前向 + PCA 投影（不计算梯度）
                with torch.no_grad():
                    t_a = project_teacher(teacher.embedding_net(anchor))
                    t_p = project_teacher(teacher.embedding_net(positive))
                    t_n = project_teacher(teacher.embedding_net(negative))

                # 学生前向（输出维度 = embedding_dim，与 PCA 投影后一致）
                s_a = student_emb(anchor)
                s_p = student_emb(positive)
                s_n = student_emb(negative)

                # 三元组损失
                trip_loss = triplet_loss_fn(s_a, s_p, s_n)

                # KL 蒸馏损失（对三个 embedding 分别计算）
                kd_loss = (
                    F.kl_div(F.log_softmax(s_a / temperature, dim=1),
                             F.softmax(t_a / temperature, dim=1),
                             reduction='batchmean') +
                    F.kl_div(F.log_softmax(s_p / temperature, dim=1),
                             F.softmax(t_p / temperature, dim=1),
                             reduction='batchmean') +
                    F.kl_div(F.log_softmax(s_n / temperature, dim=1),
                             F.softmax(t_n / temperature, dim=1),
                             reduction='batchmean')
                ) * (temperature ** 2)

                loss = (1 - alpha) * trip_loss + alpha * kd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            loss_ep = total_loss / len(train_loader) * 10
            loss_per_epoch.append(loss_ep)

            # 验证：在 embedding 上做 5-NN 分类
            student_emb.eval()
            with torch.no_grad():
                valid_emb = student_emb(valid_data_tensor)
                dist_mat = torch.cdist(valid_emb, valid_emb, p=2)
                _, sorted_idx = torch.sort(dist_mat, dim=1)
                k_idx = sorted_idx[:, 1:6]  # 排除自身
                neigh_labels = valid_labels_tensor[k_idx]
                predicted = torch.mode(neigh_labels, dim=1)[0]
                accuracy = (predicted == valid_labels_tensor).float().mean().item() * 100

            # 检查是否改进
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                best_state_dict = {k: v.cpu().clone() for k, v in student_emb.state_dict().items()}
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1

            total_bar.update(1)
            
            # 显示进度信息
            current_epoch = epoch + 1
            can_early_stop = current_epoch >= min_epochs  # 只有达到最小epoch后才能早停
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                postfix_dict = {
                    'loss': f'{loss_ep:.4f}',
                    'acc': f'{accuracy:.1f}%',
                    'best': f'{best_accuracy:.1f}%',
                }
                if can_early_stop:
                    postfix_dict['no_imp'] = f'{no_improve_count}/{patience}'
                else:
                    postfix_dict['warmup'] = f'{current_epoch}/{min_epochs}'
                total_bar.set_postfix(postfix_dict)

            # 早停检查（仅在达到最小epoch后生效）
            if can_early_stop and no_improve_count >= patience:
                print(f"\n✅ 早停触发！已连续 {patience} 个epoch准确率未提升")
                print(f"   最佳准确率: {best_accuracy:.2f}% (第 {best_epoch} 轮)")
                print(f"   当前epoch: {current_epoch}/{num_epochs}")
                print(f"   已满足最小epoch要求 ({min_epochs})")
                break
            
            # 如果未达到最小epoch，显示提示
            if not can_early_stop and no_improve_count >= patience:
                total_bar.set_description(f"蒸馏进度 (warmup {current_epoch}/{min_epochs})")

    # 保存最佳学生模型（同时保存架构配置以便后续加载）
    model_path = os.path.join(weights_dir, "Extractor_best.pth")
    save_dict = {
        'state_dict': best_state_dict,
        'channels': channels,
        'embedding_dim': embedding_dim,
        'net_type': net_type_str,
        'width_multiplier': width_mult,
    }
    torch.save(save_dict, model_path)
    print(f"\n蒸馏完成。最佳验证准确率: {best_accuracy:.2f}%")
    print(f"学生模型已保存: {model_path}")
    return model_path


# ────────────────────────────────────────────────────────────
# 步骤 5: 评估对比
# ────────────────────────────────────────────────────────────

def step5_evaluate(run_dir, large_model_path, pruned_model_path,
                   net_type, preprocess_type, data, labels):
    """用 KNN 分类器评估两个网络，对比准确率和参数量。"""
    print("\n" + "=" * 60)
    print("  步骤 5: 评估与对比")
    print("=" * 60)

    # 划分注册集 / 测试集（按标签分层，确保每类设备在两边都有样本）
    enrol_data, test_data, enrol_labels, test_labels = train_test_split(
        data, labels, test_size=0.5, stratify=labels.ravel(), shuffle=True)

    eval_batch = 128
    results = {}

    # 释放蒸馏阶段占用的 GPU 内存
    torch.cuda.empty_cache()

    # ── 评估大网络（智能加载） ──
    print("\n--- 大网络 ---")
    large_model, is_pruned_teacher = _load_model_smart(
        large_model_path, net_type, preprocess_type)
    
    if is_pruned_teacher:
        print(f"  注意：教师网络是剪枝模型")
    
    large_model.to(DEVICE)
    large_model.eval()

    enrol_emb_large = _extract_embeddings(large_model.embedding_net, enrol_data,
                                           eval_batch, DEVICE)
    test_emb_large = _extract_embeddings(large_model.embedding_net, test_data,
                                          eval_batch, DEVICE)

    test_labels_t = torch.tensor(test_labels, dtype=torch.long).to(DEVICE)
    acc_large = _knn_accuracy(test_emb_large, enrol_emb_large,
                               enrol_labels, test_labels_t)
    n_params_large = sum(p.numel() for p in large_model.parameters())
    results['large'] = {'accuracy': acc_large, 'params': n_params_large}
    print(f"  准确率: {acc_large:.2f}%, 参数量: {n_params_large:,}")

    # ── 评估剪枝网络 ──
    print("\n--- 剪枝网络 ---")
    saved = torch.load(pruned_model_path, weights_only=True)
    channels = saved.get('channels', None)
    embedding_dim = saved.get('embedding_dim', 8)
    net_type_str = net_type.value if hasattr(net_type, 'value') else str(net_type)

    if channels is not None:
        pruned_emb = build_pruned_embedding_net(
            net_type_str, preprocess_type.in_channels, channels, embedding_dim)
    else:
        width_mult = saved.get('width_multiplier', 1 / 16)
        _, prune_cls = NET_PAIRS[net_type_str]
        if net_type_str in ('MobileNetV1', 'MobileNetV2', 'LightNet'):
            pruned_emb = prune_cls(preprocess_type.in_channels,
                                    width_multiplier=width_mult)
        else:
            pruned_emb = prune_cls(preprocess_type.in_channels)

    pruned_emb.load_state_dict(saved['state_dict'])
    pruned_emb.to(DEVICE)
    pruned_emb.eval()

    enrol_emb_pruned = _extract_embeddings(pruned_emb, enrol_data, eval_batch, DEVICE)
    test_emb_pruned = _extract_embeddings(pruned_emb, test_data, eval_batch, DEVICE)

    acc_pruned = _knn_accuracy(test_emb_pruned, enrol_emb_pruned,
                                enrol_labels, test_labels_t)
    n_params_pruned = sum(p.numel() for p in pruned_emb.parameters())
    results['pruned'] = {'accuracy': acc_pruned, 'params': n_params_pruned}
    print(f"  准确率: {acc_pruned:.2f}%, 参数量: {n_params_pruned:,}")

    # ── 对比 ──
    print("\n" + "-" * 50)
    acc_drop = results['large']['accuracy'] - results['pruned']['accuracy']
    param_ratio = results['pruned']['params'] / results['large']['params'] * 100
    print(f"  准确率变化:   {acc_drop:+.2f}%")
    print(f"  参数占比:     {param_ratio:.1f}%")
    print(f"  压缩比:       {results['large']['params'] / results['pruned']['params']:.1f}x")
    print("-" * 50)

    return results


def _extract_embeddings(embedding_net, data, batch_size, device):
    """分批提取 embedding，避免 GPU OOM"""
    embeddings = []
    for i in range(0, len(data), batch_size):
        batch = torch.tensor(data[i:i + batch_size], dtype=torch.float32).to(device)
        with torch.no_grad():
            emb = embedding_net(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).to(device)


def _knn_accuracy(test_emb, enrol_emb, enrol_labels, test_labels_t, k=5):
    """K-NN 分类准确率"""
    dist_mat = torch.cdist(test_emb, enrol_emb, p=2)
    _, sorted_idx = torch.sort(dist_mat, dim=1)
    k_idx = sorted_idx[:, :k]
    enrol_labels_t = torch.tensor(enrol_labels, dtype=torch.long).to(DEVICE)
    neigh_labels = enrol_labels_t[k_idx]
    predicted = torch.mode(neigh_labels, dim=1)[0]
    return (predicted == test_labels_t).float().mean().item() * 100


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

def run_pipeline(net_type=NetworkType.ResNet,
                 preprocess_type=PreprocessType.STFT,
                 num_epochs=50,
                 batch_size=32,
                 learning_rate=1e-3,
                 alpha=0.7,
                 temperature=3.0,
                 embedding_dim=8,
                 min_channels=2,
                 pca_threshold=0.95,
                 snr=None):
    """运行完整流水线。

    Args:
        net_type: 大网络架构
        preprocess_type: 数据预处理方式（STFT / WST / IQ）
        num_epochs: 训练和蒸馏的 epoch 数
        batch_size: 所有阶段的 batch size
        learning_rate: Adam 学习率
        alpha: 蒸馏损失权重（0 = 仅三元组，1 = 仅 KL）
        temperature: 蒸馏温度
        embedding_dim: 小网络输出 embedding 维度
        min_channels: PCA 推导通道数的下限
        pca_threshold: 有效维度的累积方差阈值
        snr: AWGN 的 SNR 范围（None = 不加噪）
    """
    set_seed(42)

    hp = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'temperature': temperature,
        'embedding_dim': embedding_dim,
        'min_channels': min_channels,
        'pca_threshold': pca_threshold,
        'snr': snr,
        'pca_max_batches': 50,
        'width_multiplier': 1 / 16,
    }

    # ── 设置运行目录 ──
    pipeline_root = Path(__file__).parent
    runs_dir = pipeline_root / "runs"
    run_id = len(list(runs_dir.glob("run_*"))) + 1
    run_dir = runs_dir / f"run_{run_id:03d}_{net_type.value}"
    _ensure_dirs(run_dir, run_dir / "large" / "weights",
                 run_dir / "pruned" / "weights")

    print("=" * 60)
    print(f"  流水线: {net_type.value}")
    print(f"  运行目录: {run_dir}")
    print(f"  超参数: {hp}")
    print("=" * 60)

    # ── 加载数据 ──
    print("\n── 加载数据 ──")
    data, labels = prepare_train_data(
        new_file_flag=False,
        filename_train_prepared_data=f"train_data_{preprocess_type.value}.h5",
        path_train_data=DATASET['Train']['no_aug'].path,
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        snr_range=snr,
        generate_type=preprocess_type,
    )
    print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")

    # ── 步骤 1: 训练大网络 ──
    large_model_path = step1_train_large(
        run_dir, net_type, data, labels, preprocess_type, hp)

    # ── 步骤 2: PCA 卷积分析 ──
    pca_results = step2_pca_analysis(
        large_model_path, net_type, preprocess_type, data, labels, hp)

    # 保存 PCA 结果
    pca_save = {k: {'original_dim': v['original_dim'],
                     'effective_dim': v['effective_dim']}
                for k, v in pca_results.items()}
    with open(os.path.join(run_dir, "pca_results.json"), 'w') as f:
        json.dump(pca_save, f, indent=2, default=int)
    print(f"PCA 结果已保存到 pca_results.json")

    # ── 步骤 3+4: 构建剪枝网络 & 蒸馏 ──
    pruned_model_path = step34_distill(
        run_dir, large_model_path, pca_results,
        net_type, preprocess_type, data, labels, hp)

    # ── 步骤 5: 评估对比 ──
    eval_results = step5_evaluate(
        run_dir, large_model_path, pruned_model_path,
        net_type, preprocess_type, data, labels)

    # 保存最终结果
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n评估结果已保存到 results.json")

    print(f"\n流水线完成！所有结果保存在: {run_dir}")
    return eval_results


# ────────────────────────────────────────────────────────────
# 迭代蒸馏主函数
# ────────────────────────────────────────────────────────────

def run_iterative_distillation(net_type=NetworkType.ResNet,
                                preprocess_type=PreprocessType.STFT,
                                num_epochs=100,
                                batch_size=32,
                                learning_rate=1e-3,
                                alpha=0.7,
                                temperature=3.0,
                                embedding_dim=8,
                                min_channels=2,
                                pca_threshold=0.95,
                                snr=None,
                                accuracy_threshold=5.0,
                                max_iterations=10,
                                resume_from=None,
                                start_iteration=1,
                                early_stopping_patience=10,
                                min_distill_epochs=20):
    """运行迭代蒸馏流水线。

    流程：
      1. 训练初始大网络
      2. PCA分析并构建小网络
      3. 知识蒸馏（带最小epoch和早停机制）
      4. 评估准确率下降
      5. 如果准确率下降 < threshold，用小网络替换大网络，继续迭代
      6. 直到准确率下降超过阈值或达到最大迭代次数

    Args:
        net_type: 初始大网络架构
        preprocess_type: 数据预处理方式（STFT / WST / IQ）
        num_epochs: 每次蒸馏的最大 epoch 数
        batch_size: 所有阶段的 batch size
        learning_rate: Adam 学习率
        alpha: 蒸馏损失权重（0 = 仅三元组，1 = 仅 KL）
        temperature: 蒸馏温度
        embedding_dim: 小网络输出 embedding 维度
        min_channels: PCA 推导通道数的下限
        pca_threshold: 有效维度的累积方差阈值
        snr: AWGN 的 SNR 范围（None = 不加噪）
        accuracy_threshold: 可接受的准确率下降阈值（百分比），默认 5%
        max_iterations: 最大迭代次数，默认 10
        resume_from: 从已有的运行目录恢复（Path 或 str），加载已完成的迭代结果
        start_iteration: 起始迭代编号（当resume_from为None时忽略）
        early_stopping_patience: 早停耐心值，连续多少个epoch准确率不变就停止，默认10
        min_distill_epochs: 最小蒸馏epoch数，确保小模型有足够时间学习，默认20

    Returns:
        dict: 包含所有迭代结果的字典
    """
    set_seed(42)

    hp = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'temperature': temperature,
        'embedding_dim': embedding_dim,
        'min_channels': min_channels,
        'pca_threshold': pca_threshold,
        'snr': snr,
        'pca_max_batches': 50,
        'width_multiplier': 1 / 16,
        'early_stopping_patience': early_stopping_patience,
        'min_distill_epochs': min_distill_epochs,
    }

    # ── 设置运行目录 ──
    pipeline_root = Path(__file__).parent
    runs_dir = pipeline_root / "runs_iterative"
    
    # 检查是否从已有结果恢复
    iteration_results = []
    current_model_path = None
    current_accuracy = None
    current_params = None
    
    if resume_from is not None:
        resume_path = Path(resume_from)
        if not resume_path.is_absolute():
            # 相对路径相对于项目根目录（main.py的父目录）解析
            resume_path = Path(__file__).resolve().parent.parent / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"恢复路径不存在: {resume_path}")
        
        # 加载已有的迭代结果
        results_file = resume_path / "iterative_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                saved_data = json.load(f)
            
            iteration_results = saved_data.get('results', [])
            print(f"\n✅ 从 {resume_path} 恢复，已加载 {len(iteration_results)} 轮迭代结果")
            
            # 获取最后一轮的学生模型作为当前教师
            if iteration_results:
                last_result = iteration_results[-1]
                # 确保路径是绝对路径，不依赖 CWD
                student_rel = last_result['model_paths']['student']
                student_path = Path(student_rel)
                if not student_path.is_absolute():
                    student_path = Path(__file__).resolve().parent.parent / student_path
                current_model_path = str(student_path)
                current_accuracy = last_result['student_accuracy']
                current_params = last_result['student_params']
                print(f"   最后一轮学生模型: {current_model_path}")
                print(f"   准确率: {current_accuracy:.2f}%, 参数量: {current_params:,}")
                
                # 确定下一轮的迭代编号
                start_iteration = last_result['iteration'] + 1
                print(f"   将从第 {start_iteration} 轮继续\n")
            else:
                print("   ⚠️  警告：iterative_results.json 存在但没有已完成的迭代结果")
                print("   将尝试从头开始训练第一轮\n")
                start_iteration = 1
                current_model_path = None  # 确保为空，触发第一轮训练
        else:
            print(f"\n⚠️  警告：未找到 {results_file}")
            print("   将尝试从头开始训练第一轮\n")
            start_iteration = 1
            current_model_path = None
        
        run_dir = resume_path
    else:
        # 创建新的运行目录
        run_id = len(list(runs_dir.glob("run_*"))) + 1
        run_dir = runs_dir / f"run_{run_id:03d}_{net_type.value}_iterative"
        _ensure_dirs(run_dir)
        start_iteration = 1

    _ensure_dirs(run_dir)

    print("=" * 80)
    print(f"  迭代蒸馏流水线: {net_type.value}")
    print(f"  运行目录: {run_dir}")
    print(f"  超参数: {hp}")
    print(f"  准确率阈值: {accuracy_threshold}%")
    print(f"  最大迭代次数: {max_iterations}")
    if resume_from:
        print(f"  恢复模式: 是 (从第 {start_iteration} 轮继续)")
    print("=" * 80)

    # ── 加载数据 ──
    print("\n── 加载数据 ──")
    data, labels = prepare_train_data(
        new_file_flag=False,
        filename_train_prepared_data=f"train_data_{preprocess_type.value}.h5",
        path_train_data=DATASET['Train']['no_aug'].path,
        dev_range=np.arange(0, 40, dtype=int),
        pkt_range=np.arange(0, 800, dtype=int),
        snr_range=snr,
        generate_type=preprocess_type,
    )
    print(f"数据形状: {data.shape}, 标签形状: {labels.shape}")

    # ── 迭代蒸馏循环 ──
    teacher_is_pruned = (len(iteration_results) > 0)  # 如果有历史结果，说明教师已经是剪枝模型

    for iteration in range(start_iteration, max_iterations + 1):
        print("\n" + "=" * 80)
        print(f"  迭代 {iteration}/{max_iterations}")
        print("=" * 80)

        iter_dir = run_dir / f"iteration_{iteration:02d}"
        _ensure_dirs(iter_dir, iter_dir / "teacher" / "weights",
                     iter_dir / "student" / "weights")

        # ── 步骤 1: 训练/加载教师网络 ──
        if current_model_path is None:
            # 没有上一轮结果，需要从头训练大网络
            print(f"\n[迭代 {iteration}] 训练初始大网络...")
            large_model_path = step1_train_large(
                iter_dir, net_type, data, labels, preprocess_type, hp)
            current_model_path = large_model_path
            teacher_is_pruned = False
        else:
            # 有上一轮结果，使用上一轮的学生网络作为教师
            print(f"\n[迭代 {iteration}] 使用上一轮学生网络作为教师...")
            large_model_path = current_model_path
            teacher_is_pruned = True

        # ── 步骤 2: PCA 卷积分析 ──
        print(f"\n[迭代 {iteration}] PCA 卷积层冗余分析...")
        pca_results = step2_pca_analysis(
            large_model_path, net_type, preprocess_type, data, labels, hp)

        # 保存 PCA 结果
        pca_save = {k: {'original_dim': v['original_dim'],
                         'effective_dim': v['effective_dim']}
                    for k, v in pca_results.items()}
        with open(os.path.join(iter_dir, "pca_results.json"), 'w') as f:
            json.dump(pca_save, f, indent=2, default=int)

        # ── 步骤 3+4: 构建剪枝网络 & 蒸馏 ──
        print(f"\n[迭代 {iteration}] 构建剪枝网络并进行蒸馏...")
        pruned_model_path = step34_distill(
            iter_dir, large_model_path, pca_results,
            net_type, preprocess_type, data, labels, hp)

        # ── 步骤 5: 评估对比 ──
        print(f"\n[迭代 {iteration}] 评估与对比...")
        eval_results = step5_evaluate(
            iter_dir, large_model_path, pruned_model_path,
            net_type, preprocess_type, data, labels)

        # ── 检查准确率下降 ──
        acc_teacher = eval_results['large']['accuracy']
        acc_student = eval_results['pruned']['accuracy']
        params_teacher = eval_results['large']['params']
        params_student = eval_results['pruned']['params']
        acc_drop = acc_teacher - acc_student
        param_ratio = params_student / params_teacher * 100
        compression_ratio = params_teacher / params_student

        print(f"\n{'='*60}")
        print(f"[迭代 {iteration}] 结果汇总:")
        print(f"  教师准确率: {acc_teacher:.2f}%")
        print(f"  学生准确率: {acc_student:.2f}%")
        print(f"  准确率下降: {acc_drop:.2f}%")
        print(f"  教师参数量: {params_teacher:,}")
        print(f"  学生参数量: {params_student:,}")
        print(f"  参数占比:   {param_ratio:.2f}%")
        print(f"  压缩比:     {compression_ratio:.2f}x")
        print(f"{'='*60}")

        # 记录本轮结果
        iter_result = {
            'iteration': iteration,
            'teacher_accuracy': acc_teacher,
            'student_accuracy': acc_student,
            'accuracy_drop': acc_drop,
            'teacher_params': params_teacher,
            'student_params': params_student,
            'param_ratio': param_ratio,
            'compression_ratio': compression_ratio,
            'teacher_is_pruned': teacher_is_pruned,
            'model_paths': {
                'teacher': str(large_model_path),
                'student': str(pruned_model_path),
            }
        }
        iteration_results.append(iter_result)

        # ── 判断是否继续迭代 ──
        if acc_drop > accuracy_threshold:
            print(f"\n⚠️  准确率下降 ({acc_drop:.2f}%) 超过阈值 ({accuracy_threshold}%)，停止迭代")
            break
        else:
            print(f"\n✅ 准确率下降 ({acc_drop:.2f}%) 在可接受范围内，继续下一轮迭代")
            # 更新当前模型为学生模型，用于下一轮
            current_model_path = pruned_model_path
            current_accuracy = acc_student
            current_params = params_student
            teacher_is_pruned = True

        # 每轮结束后保存中间结果
        final_summary = {
            'total_iterations': len(iteration_results),
            'accuracy_threshold': accuracy_threshold,
            'max_iterations': max_iterations,
            'resume_from': str(resume_from) if resume_from else None,
            'results': iteration_results,
        }
        with open(os.path.join(run_dir, "iterative_results.json"), 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)

    # ── 保存最终结果 ──
    final_summary = {
        'total_iterations': len(iteration_results),
        'accuracy_threshold': accuracy_threshold,
        'max_iterations': max_iterations,
        'resume_from': str(resume_from) if resume_from else None,
        'results': iteration_results,
    }
    with open(os.path.join(run_dir, "iterative_results.json"), 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("  迭代蒸馏完成！")
    print("=" * 80)
    print(f"总迭代次数: {len(iteration_results)}")
    if iteration_results:
        first = iteration_results[0]
        last = iteration_results[-1]
        print(f"初始教师准确率: {first['teacher_accuracy']:.2f}%")
        print(f"最终学生准确率: {last['student_accuracy']:.2f}%")
        print(f"总体准确率下降: {first['teacher_accuracy'] - last['student_accuracy']:.2f}%")
        print(f"初始参数量: {first['teacher_params']:,}")
        print(f"最终参数量: {last['student_params']:,}")
        print(f"总体压缩比: {first['teacher_params'] / last['student_params']:.2f}x")
    print(f"所有结果保存在: {run_dir}")
    print("=" * 80)

    return final_summary


# ────────────────────────────────────────────────────────────
# 命令行接口
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LoRa RFFI 轻量化流水线")
    
    # 模式选择
    ap.add_argument('--mode', type=str, default='single',
                    choices=['single', 'iterative'],
                    help='运行模式: single=单次蒸馏, iterative=迭代蒸馏')
    
    ap.add_argument('--net', type=str, default='ResNet',
                    choices=['ResNet', 'SCSKNet', 'GoogleNet', 'DenseNet', 'ShuffleNet',
                             'MobileNetV1', 'MobileNetV2', 'LightNet'],
                    help='大网络架构')
    ap.add_argument('--epochs', type=int, default=300,
                    help='训练/蒸馏 epoch 数')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3, help='学习率')
    ap.add_argument('--alpha', type=float, default=0.7,
                    help='蒸馏损失权重 (0=仅三元组, 1=仅KL)')
    ap.add_argument('--temperature', type=float, default=3.0)
    ap.add_argument('--embedding-dim', type=int, default=8,
                    help='小网络 embedding 维度')
    ap.add_argument('--min-channels', type=int, default=2,
                    help='每层卷积的最小通道数')
    ap.add_argument('--pca-threshold', type=float, default=0.95,
                    help='PCA 有效维度的累积方差阈值')
    ap.add_argument('--snr-low', type=float, default=None,
                    help='AWGN 低 SNR (dB)')
    ap.add_argument('--snr-high', type=float, default=None,
                    help='AWGN 高 SNR (dB)')
    
    # 迭代蒸馏专用参数
    ap.add_argument('--accuracy-threshold', type=float, default=80.0,
                    help='[迭代模式] 可接受的准确率下降阈值（百分比）')
    ap.add_argument('--max-iterations', type=int, default=30,
                    help='[迭代模式] 最大迭代次数')
    ap.add_argument('--resume-from', type=str, default=None,
                    help='[迭代模式] 从已有的运行目录恢复，指定路径')
    ap.add_argument('--patience', type=int, default=20,
                    help='[迭代模式] 早停耐心值，连续多少个epoch准确率不变就停止')
    ap.add_argument('--min-epochs', type=int, default=20,
                    help='[迭代模式] 最小蒸馏epoch数，确保小模型有足够时间学习')

    args = ap.parse_args()

    snr = None
    if args.snr_low is not None and args.snr_high is not None:
        snr = np.arange(args.snr_low, args.snr_high)

    if args.mode == 'iterative':
        # 运行迭代蒸馏
        run_iterative_distillation(
            net_type=NetworkType[args.net],
            preprocess_type=PreprocessType.STFT,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            alpha=args.alpha,
            temperature=args.temperature,
            embedding_dim=args.embedding_dim,
            min_channels=args.min_channels,
            pca_threshold=args.pca_threshold,
            snr=snr,
            accuracy_threshold=args.accuracy_threshold,
            max_iterations=args.max_iterations,
            resume_from=args.resume_from,
            early_stopping_patience=args.patience,
            min_distill_epochs=args.min_epochs,
        )
    else:
        # 运行单次蒸馏（原有流程）
        run_pipeline(
            net_type=NetworkType[args.net],
            preprocess_type=PreprocessType.STFT,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            alpha=args.alpha,
            temperature=args.temperature,
            embedding_dim=args.embedding_dim,
            min_channels=args.min_channels,
            pca_threshold=args.pca_threshold,
            snr=snr,
        )
