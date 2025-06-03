import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, shortcut=False):
        super().__init__()
        self.shortcut = shortcut

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)

        if self.shortcut:
            self.conv3_for_shortcut = nn.Conv2d(in_channels, out_channels, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.relu(fx)
        fx = self.conv2(fx)

        if self.shortcut:
            x = self.conv3_for_shortcut(x)

        out = x + fx
        out = self.relu(out)
        return out


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Linear(in_channels, in_channels // reduction_ratio), nn.ReLU(), nn.Linear(in_channels // reduction_ratio, in_channels))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_weights = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * channel_weights


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_weights


# 组合CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# 修改后的特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, pool_size=(7, 7), pkt_desc_vec_len=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, shortcut=True),
            ResBlock(64, 64),
            nn.AdaptiveAvgPool2d(pool_size),
            CBAM(in_channels=64, reduction_ratio=16, kernel_size=7),
            nn.Flatten(),
            nn.Linear(in_features=64 * pool_size[0] * pool_size[1], out_features=pkt_desc_vec_len),
        )

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # channels_last -> channels_first
        x = self.layers(x)
        out = F.normalize(x, p=2, dim=1)  # L2 normalization
        return out


class TripletNet(nn.Module):
    def __init__(self, in_channels=1, pool_size=(7, 7), pkt_desc_vec_len=512):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channels, pool_size, pkt_desc_vec_len)

    def forward(self, anchor, positive, negative):
        anchor_desc_vec = self.feature_extractor(anchor)
        positive_desc_vec = self.feature_extractor(positive)
        negative_desc_vec = self.feature_extractor(negative)
        return anchor_desc_vec, positive_desc_vec, negative_desc_vec


# 使用示例
if __name__ == "__main__":
    ################################################
    # Check the model parameters
    ################################################
    input_shape = (32, 102, 62, 1)
    feature_extractor = FeatureExtractor(in_channels=1)
    stat(feature_extractor, (102, 62, 1))

    # ----------------------------------------------
    # [Output]:
    #   Total params: 12,458,496
    # ----------------------------------------------
    # Correspond to the trainable params (12,458,496) in paper
    # ----------------------------------------------
