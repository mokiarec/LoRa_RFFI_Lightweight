import h5py
import numpy as np
import pickle
from utils.signal_trans import TimeFrequencyTransformer, awgn
from core.config import PreprocessType
import os

def process_iq_dataset_to_pickle(h5_file_path, output_dir, devices_per_class=200,
                                preprocess_type=PreprocessType.STFT, snr_range=None):
    """
    处理IQ数据集，转换为pickle格式，每个设备200个包

    Args:
        h5_file_path: 输入的h5文件路径
        output_dir: 输出目录
        devices_per_class: 每个设备的包数量
        preprocess_type: 预处理类型
        snr_range: 信噪比范围
    """
    os.makedirs(output_dir, exist_ok=True)

    # 读取H5文件
    with h5py.File(h5_file_path, 'r') as f:
        data = f['data'][:]
        labels = f['label'][:]

    print(f"原始数据形状: {data.shape}")
    print(f"原始标签形状: {labels.shape}")

    # 修正标签形状，确保是一维数组
    if labels.ndim > 1:
        labels = labels.flatten()  # 将多维标签转为一维

    print(f"修正后标签形状: {labels.shape}")

    # 获取每个设备的包数量
    unique_labels = np.unique(labels)
    print(f"设备数量: {len(unique_labels)}")
    print(f"设备ID: {unique_labels}")

    processed_data = []
    processed_labels = []

    for device_id in unique_labels:
        # 获取当前设备的所有数据
        device_mask = (labels == device_id)
        device_data = data[device_mask]

        print(f"设备 {device_id} 原始包数量: {len(device_data)}")

        # 如果包数量超过200，截取前200个
        if len(device_data) > devices_per_class:
            device_data = device_data[:devices_per_class]
        # 如果包数量不足200，重复数据以达到200个
        elif len(device_data) < devices_per_class:
            repeat_times = devices_per_class // len(device_data)
            remainder = devices_per_class % len(device_data)
            device_data = np.concatenate([np.tile(device_data, (repeat_times, 1)),
                                        device_data[:remainder]], axis=0)

        print(f"设备 {device_id} 处理后包数量: {len(device_data)}")

        # 添加噪声（可选）
        if snr_range is not None:
            print(f"为设备 {device_id} 添加噪声...")
            device_data = awgn(device_data, snr_range)
        else:
            print(f"设备 {device_id} 无需添加噪声")

        # 预处理
        if preprocess_type == PreprocessType.STFT:
            device_data = TimeFrequencyTransformer.generate_stft_channel(device_data)
        elif preprocess_type == PreprocessType.WST:
            device_data = TimeFrequencyTransformer.generate_WST(device_data)

        # 生成对应的标签
        device_labels = np.full(len(device_data), device_id)

        processed_data.append(device_data)
        processed_labels.append(device_labels)

    # 合并所有设备的数据
    final_data = np.concatenate(processed_data, axis=0)
    final_labels = np.concatenate(processed_labels, axis=0)

    print(f"最终数据形状: {final_data.shape}")
    print(f"最终标签形状: {final_labels.shape}")

    # 保存为pickle文件
    output_path = os.path.join(output_dir, "processed_dataset.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': final_data,
            'labels': final_labels,
            'devices_per_class': devices_per_class
        }, f)

    print(f"数据已保存到: {output_path}")
    return final_data, final_labels

def load_pickle_dataset(pickle_path):
    """
    加载pickle格式的数据集
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['data'], data['labels'], data.get('devices_per_class', 200)

# 使用示例
if __name__ == "__main__":
    # 设置参数
    h5_file_path = ".\\dataset\\Test\\dataset_residential.h5"
    output_dir = ".\\dataset\\processed"

    # 处理数据集
    processed_data, processed_labels = process_iq_dataset_to_pickle(
        h5_file_path=h5_file_path,
        output_dir=output_dir,
        devices_per_class=200,
        preprocess_type=PreprocessType.STFT,  # 或者 PreprocessType.WST
        # snr_range=[-5, 20]  # 可选的信噪比范围
    )

    # 验证处理结果
    print("数据集处理完成!")
    print(f"数据形状: {processed_data.shape}")
    print(f"标签形状: {processed_labels.shape}")

    # 检查每个设备的包数量
    unique_labels = np.unique(processed_labels)
    for label in unique_labels:
        count = np.sum(processed_labels == label)
        print(f"设备 {label}: {count} 个包")
