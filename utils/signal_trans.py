from math import ceil, floor
from matplotlib import pyplot as plt
import numpy as np
from numpy import sum, sqrt
from numpy.random import standard_normal, uniform

import scipy.signal as signal
from kymatio import Scattering1D
from tqdm import tqdm


# Additive White Gaussian Noise (AWGN) Function
def awgn(data, snr_range):
    """
    向输入数据中的每个数据包添加加性高斯白噪声（AWGN）。

    参数:
    data (numpy.ndarray): 二维数组，其中每行代表一个数据包（复数信号）。
    snr_range (tuple or list): 包含SNR范围（以dB为单位）的元组或列表。

    返回:
    numpy.ndarray: 包含添加了噪声的数据包的新数据数组。
    """
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0], snr_range[-1], pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        SNR_linear = 10 ** (SNRdB[pktIdx] / 10)
        P = sum(abs(s) ** 2) / len(s)
        N0 = P / SNR_linear
        n = sqrt(N0 / 2) * (standard_normal(len(s)) + 1j * standard_normal(len(s)))
        data[pktIdx] = s + n

    return data


class TimeFrequencyTransformer:
    """将时域信号转化为时频图，包括通道独立频谱图和小波散射频谱图的生成"""

    @staticmethod
    def generate_stft_channel(data, win_len=256, overlap=128):
        """
        将一批IQ信号样本转换为通道独立频谱图。

        参数:
        data (ndarray): 输入的IQ样本数据，每一行代表一个IQ样本。
        win_len (int, 可选): 窗口长度，默认为256。
        overlap (int, 可选): 窗口重叠长度，默认为128。

        返回:
        ndarray: 转换后的通道独立频谱图，形状为 (样本数量, 频谱图行数, 频谱图列数, 1)。
        """

        def _normalization(data):
            """对输入信号进行归一化处理"""
            s_norm = np.zeros(data.shape, dtype=complex)
            for i in range(data.shape[0]):
                sig_amplitude = np.abs(data[i])
                rms = np.sqrt(np.mean(sig_amplitude**2))
                s_norm[i] = data[i] / rms
            return s_norm

        def _spec_crop(x):
            """裁剪生成的通道独立频谱图"""
            num_row = x.shape[0]
            x_cropped = x[round(num_row * 0.3) : round(num_row * 0.7)]
            return x_cropped

        def _gen_single_channel_ind_spectrogram(sig, win_len=256, overlap=128):
            """根据设定的窗口长度和重叠长度，将IQ样本转换为单个通道独立频谱图"""
            # Short-time Fourier transform (STFT).
            f, t, spec = signal.stft(
                sig,
                window="boxcar",
                nperseg=win_len,
                noverlap=overlap,
                nfft=win_len,
                return_onesided=False,
                padded=False,
                boundary=None,
            )

            # FFT shift to adjust the central frequency.
            spec = np.fft.fftshift(spec, axes=0)

            # Generate channel independent spectrogram.
            chan_ind_spec = spec[:, 1:] / spec[:, :-1]
            # chan_ind_spec = spec

            # Take the logarithm of the magnitude.
            chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec) ** 2)

            """绘制结果，不要删除！！！"""
            # plt.figure(figsize=(8, 8))
            # plt.subplot(2, 1, 1)
            # spec_amp = np.log10(np.abs(spec) ** 2)
            # plt.imshow(spec_amp, aspect="auto",origin='lower')
            # plt.title("(a)")
            # # plt.title("STFT")
            # plt.axis('off')  # 隐藏坐标轴
            # plt.subplot(2, 1, 2)
            # plt.imshow(chan_ind_spec_amp, aspect="auto",origin='lower')
            # plt.title("(b)")
            # # plt.title("Channel independent spectrogram")
            # plt.axis('off')  # 隐藏坐标轴
            # plt.tight_layout()
            # plt.show()

            # """只打印独立频谱图"""
            # plt.figure(figsize=(8, 6))
            # plt.imshow(chan_ind_spec_amp, aspect="auto", origin='lower')
            # plt.axis('off')
            # plt.tight_layout(pad=0)  # 移除边距
            # plt.savefig('paper/figures/spec.png', dpi=600, bbox_inches='tight', pad_inches=0)  # 或 .svg
            # plt.show()
            return chan_ind_spec_amp

        # Normalize the IQ samples.
        data = _normalization(data)

        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256 * 0.4)
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, 1, num_row, num_column])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in tqdm(range(num_sample)):
            chan_ind_spec_amp = _gen_single_channel_ind_spectrogram(
                data[i], win_len, overlap
            )
            chan_ind_spec_amp = _spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, 0, :, :] = chan_ind_spec_amp

        return data_channel_ind_spec

    @staticmethod
    def generate_WST(data, J=6, Q=6):
        """
        将批量IQ样本转换为小波散射特征图。

        参数:
        data (ndarray): 输入的IQ样本数据，其中每一行代表一个IQ样本。
        J (int, 可选): 小波散射的尺度，默认为6。
        Q (int, 可选): 每倍频程的滤波器数量，默认为8。

        返回:
        ndarray: 一个三维数组，包含所有IQ样本对应的小波散射特征图。
                形状为 (样本数量, 时间点数, 特征数)。
        """
        num_sample, sample_length = data.shape

        # 初始化小波散射
        scattering = Scattering1D(J=J, Q=Q, shape=(sample_length,))

        meta = scattering.meta()
        order0 = np.where(meta["order"] == 0)
        order1 = np.where(meta["order"] == 1)[0]
        order2 = np.where(meta["order"] == 2)

        # 计算小波散射输出形状
        scatter_example = scattering(np.real(data[0]))  # 使用第一个样本计算输出
        num_row, num_column = scatter_example.shape  # 确定时间点数和特征数

        # 初始化存储矩阵
        s, e = floor(0.1 * num_row), floor(0.4 * num_row)
        data_wav_spec = np.zeros((num_sample, 2, len(order1), num_column - 1))

        for i in tqdm(range(num_sample)):
            sig = np.asarray(data[i])  # 当前样本信号

            # 分离复数信号的实部和虚部
            real_part = np.real(sig)
            imag_part = np.imag(sig)

            # 分别计算实部和虚部的小波散射
            scatter_real = scattering(real_part)
            scatter_imag = scattering(imag_part)

            # # 将实部和虚部结果组合（例如计算幅度）
            scatter_combined = np.stack((scatter_real, scatter_imag), axis=0)

            """绘制结果，不要删除！！！"""
            # for i in range(2):
            #     plt.figure(figsize=(8, 8))
            #     plt.subplot(3, 1, 1)
            #     plt.plot(scatter_combined[i][order0][0])
            #     plt.title("Zeroth-order scattering")
            #     plt.subplot(3, 1, 2)
            #     plt.imshow(scatter_combined[i][order1], aspect="auto")
            #     plt.title("First-order scattering")
            #     plt.subplot(3, 1, 3)
            #     plt.imshow(scatter_combined[i][order2], aspect="auto")
            #     plt.title("Second-order scattering")
            #     plt.tight_layout()
            #     print(
            #         scatter_combined[i][order1].shape, scatter_combined[i][order2].shape
            #     )
            #     plt.show()

            # 只传入第一阶WST结果
            scatter_combined = scatter_combined[:, order1]
            # scatter_combined = scatter_combined[:, :, 1:] / scatter_combined[:, :, :-1]

            # 存储结果
            data_wav_spec[i] = scatter_combined
        return data_wav_spec
