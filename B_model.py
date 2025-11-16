import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# 复用你原来的基础模块（ConvBlock, ResidualConvBlock, LearnableGaussianBlur, WindowAttention, SwinBlock）
# 为避免重复，我把这些模块直接复制在这里以保证每个variant独立可跑
# -------------------

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

        
class LDCTNet_Swin1(nn.Module):
    """Variant B: 高频 Swin 层数从3->1，其余保持不变"""
    def __init__(self):
        super().__init__()
        self.gaussian = LearnableGaussianBlur(1, 5, 1.5)

        self.conv_hr1 = ConvBlock(1, 32, pool=True)
        self.conv_hr2 = ConvBlock(32, 64, pool=True)
        self.conv_hr3 = ConvBlock(64, 128, pool=True)
        self.conv_hr4 = ConvBlock(128, 256, pool=True)

        self.conv_lr1 = ConvBlock(1, 32, pool=True)
        self.conv_lr2 = ConvBlock(32, 64, pool=True)
        self.conv_lr3 = ConvBlock(64, 128, pool=True)
        self.conv_lr4 = ConvBlock(128, 256, pool=True)

        # 高频仅1层 Swin（原来是3层）
        self.swin_high1 = SwinBlock(256, 4, 4)
        # 如果你想严格测试不同 depth，也可以把下面两个置为空或 identity，
        # 这里我只保留一层并不创建多余层。
        self.swin_high2 = nn.Identity()
        self.swin_high3 = nn.Identity()

        # 低频仍为1层
        self.swin_low = SwinBlock(256, 4, 4)

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
        self.out_act = nn.ReLU()

    def forward(self, x):
        x_low = self.gaussian(x)
        x_high = x - x_low

        fea_high = self.conv_hr4(self.conv_hr3(self.conv_hr2(self.conv_hr1(x_high))))
        fea_low = self.conv_lr4(self.conv_lr3(self.conv_lr2(self.conv_lr1(x_low))))

        B, C, H, W = fea_high.shape
        h_flat = fea_high.flatten(2).transpose(1, 2)
        # 仅一层 Swin
        h_feat = self.swin_high1(h_flat)
        h_feat = h_feat.transpose(1, 2).reshape(B, C, H, W)

        l_flat = fea_low.flatten(2).transpose(1, 2)
        l_feat = self.swin_low(l_flat)
        l_feat = l_feat.transpose(1, 2).reshape(B, C, H, W)

        fused = self.res_fuse(h_feat + l_feat)

        x = self.upsample1(fused)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return self.out_act(x)

# quick test
if __name__ == "__main__":
    m = LDCTNet_Swin1()
    inp = torch.randn(1,1,256,256)
    out = m(inp)
    print("Swin1 out shape:", out.shape)
