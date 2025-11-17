import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# ConvBlock + InstanceNorm
# ======================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        self.pool = pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_ch),
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
# ResidualConvBlock
# ======================
class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


# ======================
# Learnable Gaussian Blur
# ======================
class LearnableGaussianBlur(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.conv = nn.Conv2d(
            channels, channels, kernel_size, padding=kernel_size // 2,
            groups=channels, bias=False
        )
        self._init_gaussian_weights()

    def _init_gaussian_weights(self):
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
# Swin Transformer Block
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
# 主网络结构 (改进版)
# ======================
class LDCTNet_Swin_improve(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 可学习高斯核分频 ---
        self.gaussian = LearnableGaussianBlur(1, 5, 1.5)

        # 高频分支
        self.conv_hr1 = ConvBlock(1, 32, pool=True)
        self.conv_hr2 = ConvBlock(32, 64, pool=True)
        self.conv_hr3 = ConvBlock(64, 128, pool=True)
        self.conv_hr4 = ConvBlock(128, 256, pool=True)

        # 低频分支
        self.conv_lr1 = ConvBlock(1, 32, pool=True)
        self.conv_lr2 = ConvBlock(32, 64, pool=True)
        self.conv_lr3 = ConvBlock(64, 128, pool=True)
        self.conv_lr4 = ConvBlock(128, 256, pool=True)

        # Swin Transformer 主干
        # 高频路径用3层
        self.swin_high1 = SwinBlock(256, 4, 4)
        self.swin_high2 = SwinBlock(256, 4, 4)
        self.swin_high3 = SwinBlock(256, 4, 4)
        # 低频路径用1层
        self.swin_low = SwinBlock(256, 4, 4)

        # 融合模块
        self.res_fuse = ResidualConvBlock(256)

        # 上采样
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
        self.out_act = nn.ReLU()

    def forward(self, x):
        # ------------- 分频 ----------------
        x_low = self.gaussian(x)
        x_high = x - x_low

        # ------------- 特征提取 ----------------
        fea_high = self.conv_hr4(self.conv_hr3(self.conv_hr2(self.conv_hr1(x_high))))
        fea_low = self.conv_lr4(self.conv_lr3(self.conv_lr2(self.conv_lr1(x_low))))

        # ------------- Swin Transformer ----------------
        B, C, H, W = fea_high.shape
        # 高频3层 Swin
        h_flat = fea_high.flatten(2).transpose(1, 2)
        h_feat = self.swin_high3(self.swin_high2(self.swin_high1(h_flat)))
        h_feat = h_feat.transpose(1, 2).reshape(B, C, H, W)

        # 低频1层 Swin
        l_flat = fea_low.flatten(2).transpose(1, 2)
        l_feat = self.swin_low(l_flat)
        l_feat = l_feat.transpose(1, 2).reshape(B, C, H, W)

        # ------------- 高频 + 低频融合（残差式） ----------------
        fused = self.res_fuse(h_feat + l_feat)

        # ------------- 上采样到原分辨率 ----------------
        x = self.upsample1(fused)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)

        return self.out_act(x)


# # ----------------- quick smoke test -----------------
# if __name__ == "__main__":
#     dummy_input = torch.randn(1, 1, 256, 256)
#     model = LDCTNet_Swin_improve()
#     out = model(dummy_input)
#     print("input:", dummy_input.shape)
#     print("output:", out.shape)

