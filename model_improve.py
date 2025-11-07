import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ======================
# ConvBlock 增加 InstanceNorm2d
# ======================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.pool = pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),           # ✅ 加入 InstanceNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
        if pool:
            self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.avgpool(x)
        return x


# ======================
# ResidualConvBlock 同样加入 InstanceNorm
# ======================
class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# ======================
# 可学习高斯卷积模块 (替代固定高斯核)
# ======================
class LearnableGaussianBlur(nn.Module):
    """可学习的高斯滤波器，用Conv2d实现并初始化为高斯权重"""
    def __init__(self, channels=1, kernel_size=5, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,   # depthwise
            bias=False
        )
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
        """初始化为固定高斯核"""
        k = self.kernel_size
        sigma = self.sigma
        coords = torch.arange(k) - k // 2
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing="xy")
        g = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        weight = g.view(1, 1, k, k).repeat(self.conv.out_channels, 1, 1, 1)
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, x):
        return self.conv(x)


# ======================
# Swin Attention Block（略，保持原实现）
# ======================
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=4, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# SwinBlock 同样保持原有功能
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4):
        super().__init__()
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ======================
# 主网络结构 (LDCTNet_Swin)
# ======================
class LDCTNet_Swin_improve(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 可学习高斯核 ---
        self.gaussian = LearnableGaussianBlur(channels=1, kernel_size=5, sigma=1.5)

        # --- 高频与低频路径 ---
        self.conv_hr1 = ConvBlock(1, 64, pool=True)
        self.conv_hr2 = ConvBlock(64, 128, pool=True)
        self.conv_hr3 = ConvBlock(128, 256, pool=True)
        self.conv_hr4 = ConvBlock(256, 256, pool=True)

        self.conv_lr1 = ConvBlock(1, 64, pool=True)
        self.conv_lr2 = ConvBlock(64, 128, pool=True)
        self.conv_lr3 = ConvBlock(128, 256, pool=True)
        self.conv_lr4 = ConvBlock(256, 256, pool=True)

        # --- Swin Transformer 主干 ---
        self.swin_enc1 = SwinBlock(256, num_heads=4, window_size=4)
        self.swin_enc2 = SwinBlock(256, num_heads=4, window_size=4)
        self.swin_enc3 = SwinBlock(256, num_heads=4, window_size=4)

        self.swin_dec1 = SwinBlock(256, num_heads=4, window_size=4)
        self.swin_dec2 = SwinBlock(256, num_heads=4, window_size=4)
        self.swin_dec3 = SwinBlock(256, num_heads=4, window_size=4)

        # --- 融合与上采样 ---
        self.res_fuse = ResidualConvBlock(256)
        self.upsample1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample3 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        

        self.final_conv = nn.Conv2d(4, 1, 3, 1, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # ------------------- 高低频分离 -------------------
        x_low = self.gaussian(x)          # 低频
        x_high = x - x_low                # 高频

         # ------------------- HR/LR 特征提取 -------------------
        x_hr = self.conv_hr4(self.conv_hr3(self.conv_hr2(self.conv_hr1(x_high))))
        x_lr = self.conv_lr4(self.conv_lr3(self.conv_lr2(self.conv_lr1(x_low))))

    # ------------------- Swin Transformer 编码解码 -------------------
        B, C, H, W = x_lr.shape
        x_flat = x_lr.flatten(2).transpose(1, 2)   # (B, N, C)
        x_enc = self.swin_enc3(self.swin_enc2(self.swin_enc1(x_flat)))
        x_dec = self.swin_dec3(self.swin_dec2(self.swin_dec1(x_enc)))
        x_dec = x_dec.transpose(1, 2).reshape(B, C, H, W)

    # ------------------- 融合 -------------------
        fea_fused = self.res_fuse(x_dec + x_hr)

    # ------------------- 上采样 PixelShuffle 3 次 -------------------
    # fea_fused 假设 [B,256,16,16]
        x = self.upsample1(fea_fused)      # -> [B,64,32,32]
        x = self.upsample2(x)               # -> [B,16,64,64]
        x = self.upsample3(x)               # -> [B,4,128,128]
       

    # ------------------- 最终输出到256x256 -------------------
        x = F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)              # -> [B,1,256,256]

    # ------------------- 输出激活 -------------------
        return self.out_act(x)
