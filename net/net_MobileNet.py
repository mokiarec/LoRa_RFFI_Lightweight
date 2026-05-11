# net/net_mobilenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    """MobileNetV1 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(MobileNetV1, self).__init__()
        self.width_multiplier = width_multiplier
        
        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        
        def conv_dw(inp, oup, stride):
            '''深度可分离卷积块：深度卷积 + 逐点卷积'''
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        
        # 根据宽度乘数调整通道数
        output_channels = int(32 * width_multiplier)
        self.model = nn.Sequential(
            conv_bn(in_channels, output_channels, 2),
            conv_dw(output_channels, int(64*width_multiplier), 1),
            conv_dw(int(64*width_multiplier), int(128*width_multiplier), 2),
            conv_dw(int(128*width_multiplier), int(128*width_multiplier), 1),
            conv_dw(int(128*width_multiplier), int(256*width_multiplier), 2),
            conv_dw(int(256*width_multiplier), int(256*width_multiplier), 1),
            conv_dw(int(256*width_multiplier), int(512*width_multiplier), 2),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(512*width_multiplier), 1),
            conv_dw(int(512*width_multiplier), int(1024*width_multiplier), 2),
            conv_dw(int(1024*width_multiplier), int(1024*width_multiplier), 1),
        )

        # 全局平均池化层：将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层：将多维特征转换为一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
        self.fc = nn.Linear(int(1024*width_multiplier), 512)

    def forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x


class MobileNetV2(nn.Module):
    """MobileNetV2 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(MobileNetV2, self).__init__()
        self.width_multiplier = width_multiplier

        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU6'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        def conv_1x1_bn(inp, oup):
            '''1x1卷积块：1x1卷积 + 批归一化 + ReLU6'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        class InvertedResidual(nn.Module):
            '''倒残差结构'''
            def __init__(self, inp, oup, stride, expand_ratio):
                super(InvertedResidual, self).__init__()
                self.stride = stride
                assert stride in [1, 2]

                hidden_dim = round(inp * expand_ratio)
                self.use_res_connect = self.stride == 1 and inp == oup

                if expand_ratio == 1:
                    self.conv = nn.Sequential(
                        # 深度卷积
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # 逐点卷积线性变换
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(oup),
                    )
                else:
                    self.conv = nn.Sequential(
                        # 逐点卷积升维
                        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # 深度卷积
                        nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True),
                        # 逐点卷积线性变换
                        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(oup),
                    )

            def forward(self, x):
                if self.use_res_connect:
                    return x + self.conv(x)
                else:
                    return self.conv(x)

        # 构建网络
        input_channel = int(32 * width_multiplier)
        last_channel = int(1024 * width_multiplier) if width_multiplier > 1.0 else 1024

        # 网络配置：[t, c, n, s]
        # t: 扩展因子, c: 输出通道数, n: 重复次数, s: 步长
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 第一层
        self.features = [conv_bn(in_channels, input_channel, 2)]

        # 倒残差块
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel

        # 最后几层
        self.features.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*self.features)

        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
        self.fc = nn.Linear(last_channel, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2, dim=1)  # L2 正则化
        return x


class LightNet(nn.Module):
    """轻量残差 MobileNetV1 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(LightNet, self).__init__()
        self.width_multiplier = width_multiplier

        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        class ResidualDepthwiseSeparable(nn.Module):
            '''带残差连接的深度可分离卷积块'''

            def __init__(self, inp, oup, stride):
                super(ResidualDepthwiseSeparable, self).__init__()
                self.stride = stride
                self.use_res_connect = stride == 1 and inp == oup

                self.depthwise = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU6(inplace=True)
                )

                self.pointwise = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            def forward(self, x):
                if self.use_res_connect:
                    identity = x
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x + identity  # 残差连接
                else:
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x

        # 根据宽度乘数调整通道数
        output_channels = int(32 * width_multiplier)
        self.initial_conv = conv_bn(in_channels, output_channels, 2)

        # 构建带残差连接的深度可分离卷积层
        self.layers = nn.ModuleList([
            ResidualDepthwiseSeparable(output_channels, int(64*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(64*width_multiplier), int(128*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(128*width_multiplier), int(128*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(128*width_multiplier), int(256*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(256*width_multiplier), int(256*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(256*width_multiplier), int(512*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(1024*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(1024*width_multiplier), int(1024*width_multiplier), 1),
        ])

        # 全局平均池化层：将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层：将多维特征转换为一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
        self.fc = nn.Linear(int(1024*width_multiplier), 8)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2.0, dim=1)  # L2 正则化
        return x


class LightNet_prune(nn.Module):
    """轻量残差 MobileNetV1 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(LightNet_prune, self).__init__()
        self.width_multiplier = width_multiplier

        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        class ResidualDepthwiseSeparable(nn.Module):
            '''带残差连接的深度可分离卷积块'''

            def __init__(self, inp, oup, stride):
                super(ResidualDepthwiseSeparable, self).__init__()
                self.stride = stride
                self.use_res_connect = stride == 1 and inp == oup

                self.depthwise = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU6(inplace=True)
                )

                self.pointwise = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            def forward(self, x):
                if self.use_res_connect:
                    identity = x
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x + identity  # 残差连接
                else:
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x

        # 根据宽度乘数调整通道数
        output_channels = int(32 * width_multiplier)
        self.initial_conv = conv_bn(in_channels, output_channels, 2)

        # 构建带残差连接的深度可分离卷积层
        self.layers = nn.ModuleList([
            ResidualDepthwiseSeparable(output_channels, int(64*width_multiplier), 1),
            ResidualDepthwiseSeparable(int(64*width_multiplier), int(128*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(128*width_multiplier), int(128*width_multiplier), 1),

            ResidualDepthwiseSeparable(int(128*width_multiplier), int(256*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(256*width_multiplier), int(256*width_multiplier), 1),

            ResidualDepthwiseSeparable(int(256*width_multiplier), int(512*width_multiplier), 2),
            ResidualDepthwiseSeparable(int(512*width_multiplier), int(512*width_multiplier), 1),

            ResidualDepthwiseSeparable(int(512*width_multiplier), int(1024*width_multiplier), 2),
        ])

        # 全局平均池化层：将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层：将多维特征转换为一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
        self.fc = nn.Linear(int(1024*width_multiplier), 8)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2.0, dim=1)  # L2 正则化
        return x


class LightNet_opt(nn.Module):
    """轻量残差 MobileNetV1 特征提取器"""
    def __init__(self, in_channels, width_multiplier=1.0):
        super(LightNet_opt, self).__init__()
        self.width_multiplier = width_multiplier

        def conv_bn(inp, oup, stride):
            '''标准卷积块：卷积 + 批归一化 + ReLU'''
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )

        class ResidualDepthwiseSeparable(nn.Module):
            '''带残差连接的深度可分离卷积块'''

            def __init__(self, inp, oup, stride):
                super(ResidualDepthwiseSeparable, self).__init__()
                self.stride = stride
                self.use_res_connect = stride == 1 and inp == oup

                self.depthwise = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU6(inplace=True)
                )

                self.pointwise = nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            def forward(self, x):
                if self.use_res_connect:
                    identity = x
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x + identity  # 残差连接
                else:
                    x = self.depthwise(x)
                    x = self.pointwise(x)
                    return x

        # 根据宽度乘数调整通道数
        output_channels = int(32 * width_multiplier)
        self.initial_conv = conv_bn(in_channels, output_channels, 2)

        # 构建带残差连接的深度可分离卷积层
        self.layers = nn.ModuleList([
            ResidualDepthwiseSeparable(output_channels, int(4), 1),
            ResidualDepthwiseSeparable(int(4), int(8), 2),
            ResidualDepthwiseSeparable(int(8), int(8), 1),
            ResidualDepthwiseSeparable(int(8), int(16), 2),
            ResidualDepthwiseSeparable(int(16), int(16), 1),
            ResidualDepthwiseSeparable(int(16), int(20), 2),
            ResidualDepthwiseSeparable(int(20), int(20), 1),
            ResidualDepthwiseSeparable(int(20), int(20), 1),
            ResidualDepthwiseSeparable(int(20), int(20), 1),
            ResidualDepthwiseSeparable(int(20), int(20), 1),
            ResidualDepthwiseSeparable(int(20), int(20), 1),
            ResidualDepthwiseSeparable(int(20), int(32), 2),
            ResidualDepthwiseSeparable(int(32), int(32), 1),
        ])

        # 全局平均池化层：将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 展平层：将多维特征转换为一维向量
        self.flatten = nn.Flatten()
        # 全连接层：将特征映射到512维特征空间
        self.fc = nn.Linear(int(32), 8)

    def forward(self, x):
        x = self.initial_conv(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = F.normalize(self.fc(x), p=2.0, dim=1)  # L2 正则化
        return x


def MobileNet(version, in_channels, width_multiplier=1.0):
    """返回 MobileNet 对象"""

    if version == 'v1':
        return MobileNetV1(in_channels, width_multiplier)
    if version == 'v2':
        return MobileNetV2(in_channels, width_multiplier)
    if version == 'light':
        return LightNet(in_channels, width_multiplier)
    if version == 'light_prune':
        return LightNet_prune(in_channels, width_multiplier)
    if version == 'light_opt':
        return LightNet_opt(in_channels, width_multiplier)