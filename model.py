<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


# --------------------------- 通用模块封装（减少冗余，提升可维护性）---------------------------
class ConvBlock(nn.Module):
    """通用卷积块：Conv2d + LeakyReLU，支持可选平均池化（减少高频损耗）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(inplace=True)
        self.use_pool = use_pool
        self.pool = nn.AvgPool2d(2, stride=2)  # 平均池化替代卷积步长下采样

    def forward(self, x):
        x = self.act(self.conv(x))
        if self.use_pool:
            x = self.pool(x)
        return x


class ResidualConvBlock(nn.Module):
    """残差卷积块：2个ConvBlock + 残差连接，增强特征融合能力"""
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        # 保证两个 conv 的输入输出通道一致
        self.conv1 = ConvBlock(channels, channels, kernel_size, padding=padding)
        self.conv2 = ConvBlock(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + shortcut


# --------------------------- Swin-Transformer 基础模块（优化尺寸适配）---------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # tuple (H_win, W_win)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # (2, H_win, W_win)
        coords_flatten = torch.flatten(coords, 1)  # (2, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.shift_size = shift_size

        H, W = input_resolution
        self.window_size = min(window_size, H, W)
        while H % self.window_size != 0 or W % self.window_size != 0:
            self.window_size -= 1
        if min(H, W) <= self.window_size:
            self.shift_size = 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, window_size=(self.window_size, self.window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

        if self.shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def window_partition(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows

    def window_reverse(self, windows, H, W):
        nW = H // self.window_size * W // self.window_size
        B = int(windows.shape[0] / nW)
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"输入序列长度{L}与分辨率{(H,W)}不匹配"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = self.window_partition(shifted_x)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + shortcut

        return x


# --------------------------- 主模型：LDCTNet_Swin（优化高频保留与参数适配）---------------------------
class LDCTNet_Swin(nn.Module):

    def _create_gaussian_kernel(self, kernel_size=11, sigma=1.5):
         coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
         x, y = torch.meshgrid(coords, coords, indexing='ij')
         kernel_2d = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
         kernel_2d /= kernel_2d.sum()

         # 扩展维度为 (1,1,k,k)
         kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)
         return kernel_4d



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


    def __init__(self, input_size=(256, 256), base_channels=16, swin_window_size=7, swin_num_heads=8):
        super(LDCTNet_Swin, self).__init__()
        self.input_size = input_size
        self.base_channels = base_channels
        self.swin_dim = base_channels * 16  # Swin模块输入通道数（256）
        self.swin_input_res = (16, 16)      # Swin模块输入特征图尺寸（16x16）

        # 低频路径（下采样 4 次：256→16）
        self.lr_path = nn.Sequential(
            ConvBlock(1, base_channels*1, kernel_size=5, padding=2, use_pool=True),  # 256→128
            ConvBlock(base_channels*1, base_channels*2, kernel_size=5, padding=2, use_pool=True),  # 128→64
            ConvBlock(base_channels*2, base_channels*4, kernel_size=5, padding=2, use_pool=True),  # 64→32
            ConvBlock(base_channels*4, self.swin_dim, kernel_size=5, padding=2, use_pool=True)     # 32→16
        )

        # 高频路径（保持一致，最终降采样至16×16）
        self.hr_conv1 = ConvBlock(1, base_channels*4, kernel_size=5, padding=2)
        self.hr_conv2 = ConvBlock(base_channels*4, base_channels*8, kernel_size=5, padding=2)
        self.hr_conv3 = ConvBlock(base_channels*8, base_channels*16, kernel_size=5, padding=2)
        self.hr_conv4 = ConvBlock(base_channels*16, self.swin_dim, kernel_size=5, padding=2)

        self.hr_pool1 = nn.AvgPool2d(2, stride=2)
        self.hr_pool2 = nn.AvgPool2d(2, stride=2)
        self.hr_pool3 = nn.AvgPool2d(2, stride=2)
        self.hr_pool4 = nn.AvgPool2d(2, stride=2)  # 最终 256→16

        # SwinTransformer 编码器和解码器保持不变
        self.swin_encoder = nn.ModuleList([
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=0),
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=swin_window_size // 2),
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=0)
        ])

        self.swin_decoder = nn.ModuleList([
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=swin_window_size // 2),
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=0),
            SwinBlock(self.swin_dim, self.swin_input_res, swin_num_heads, swin_window_size, shift_size=swin_window_size // 2)
        ])

        # 特征融合与上采样模块（从16×16恢复到256×256）
        self.combine_block1 = ResidualConvBlock(channels=self.swin_dim)
        self.upsample1 = nn.PixelShuffle(2)  # 16→32
        self.combine_block2 = ResidualConvBlock(channels=base_channels*4)
        self.upsample2 = nn.PixelShuffle(4)  # 32→256
        self.final_conv = nn.Conv2d(base_channels*4, 1, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size=11, sigma=1.5)
        self.register_buffer("gaussian_kernel_buf", self.gaussian_kernel, persistent=False)


        # ---------- LR path: 改为 ModuleList，便于拿到中间尺度特征 ----------
        # 512 -> 256 -> 128 -> 64 -> 32
        self.lr_blocks = nn.ModuleList([
            ConvBlock(1, base_channels * 1, kernel_size=5, padding=2, use_pool=True),   # -> 256x256, C=16
            ConvBlock(base_channels * 1, base_channels * 2, kernel_size=5, padding=2, use_pool=True),  # ->128x128, C=32
            ConvBlock(base_channels * 2, base_channels * 4, kernel_size=5, padding=2, use_pool=True),  # ->64x64, C=64
            ConvBlock(base_channels * 4, self.swin_dim, kernel_size=5, padding=2, use_pool=True)       # ->32x32, C=256
        ])

        # ---------- HR path: 修改为 clear 的 conv+pool 流程 ----------
        self.hr_conv1 = ConvBlock(1, base_channels * 4, kernel_size=5, padding=2)  # (B,64,512,512)
        self.hr_pool1 = nn.AvgPool2d(2, stride=2)  # ->256
        self.hr_conv2 = ConvBlock(base_channels * 4, base_channels * 8, kernel_size=5, padding=2)  # (B,128,256,256)
        self.hr_pool2 = nn.AvgPool2d(2, stride=2)  # ->128
        self.hr_conv3 = ConvBlock(base_channels * 8, base_channels * 16, kernel_size=5, padding=2)  # (B,256,128,128)
        self.hr_pool3 = nn.AvgPool2d(2, stride=2)  # ->64
        self.hr_conv4 = ConvBlock(base_channels * 16, self.swin_dim, kernel_size=5, padding=2)  # (B,256,64,64)
        self.hr_pool4 = nn.AvgPool2d(2, stride=2)  # ->32

        # ---------- Swin Encoder / Decoder ----------
        self.swin_encoder = nn.ModuleList([
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=0),
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=swin_window_size // 2),
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=0)
        ])

        self.swin_decoder = nn.ModuleList([
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=swin_window_size // 2),
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=0),
            SwinBlock(dim=self.swin_dim, input_resolution=self.swin_input_res, num_heads=swin_num_heads,
                      window_size=swin_window_size, shift_size=swin_window_size // 2)
        ])

        # 特征融合与上采样
        self.combine_block1 = ResidualConvBlock(channels=self.swin_dim)    # work on 256-ch (32x32)
        # 第一次上采样：PixelShuffle(2) 需要 in_channels = out_channels * (2^2)
        # 我们的 in_channels = 256 -> out_channels = 256/4 = 64, H/W: 32->64
        self.upsample1 = nn.PixelShuffle(2)

        # combine_block2 operates on 64 channels at 64x64
        self.combine_block2 = ResidualConvBlock(channels=base_channels * 4)  # channels=64

        # 第二次上采样：PixelShuffle(8) 需要 in_channels = out_channels * (8^2)
        # 我们希望最终 out_channels=1 -> in_channels must be 64 (=1*64) -> matches combine_block2 out channels
        self.upsample2 = nn.PixelShuffle(8)

        # final conv: after upsample2, channels will be 1, 所以 final_conv 输入应为 1
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    



    def forward(self, x):
        """
        x: (B,1,H,W) assumed H=W=512
        returns out: (B,1,H,W)
        """
        # 1. 高低频分离
        img_LR = F.conv2d(x, self.gaussian_kernel_buf.to(x.device), padding=5)  # (B,1,512,512)
        img_HR = x - img_LR

        # 2. LR path: 逐层计算并保存 64x64 的中间特征
        lr = img_LR
        lr_feats = []
        for idx, blk in enumerate(self.lr_blocks):
            lr = blk(lr)
            lr_feats.append(lr)  # lr_feats[0]: 256x256 (C=16); lr_feats[1]:128x128(C=32); lr_feats[2]:64x64(C=64); lr_feats[3]:32x32(C=256)
        x_lr_32 = lr_feats[-1]      # (B,256,32,32)
        x_lr_64 = lr_feats[-2]      # (B,64,64,64)  <-- 用于中间融合

        # 3. HR path: 清晰且正确的下采样序列
        x_hr = self.hr_conv1(img_HR)    # (B,64,512,512)
        x_hr = self.hr_pool1(x_hr)      # (B,64,256,256)
        x_hr = self.hr_conv2(x_hr)      # (B,128,256,256)
        x_hr = self.hr_pool2(x_hr)      # (B,128,128,128)
        x_hr = self.hr_conv3(x_hr)      # (B,256,128,128)
        x_hr = self.hr_pool3(x_hr)      # (B,256,64,64)
        x_hr = self.hr_conv4(x_hr)      # (B,256,64,64)
        x_hr = self.hr_pool4(x_hr)      # (B,256,32,32)  final HR feature map to feed Swin decoder

        # 4. Swin Encoder on low-frequency feature x_lr_32 (B,256,32,32)
        B, C, H, W = x_lr_32.shape
        x_swin = x_lr_32.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.swin_encoder:
            x_swin = blk(x_swin)
        x_swin_enc = x_swin.transpose(1, 2).view(B, C, H, W)  # (B,256,32,32)

        # 5. Swin Decoder: 把 HR 特征与 Encoder 输出融合（以序列形式）
        x_swin_hr = x_hr.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_swin_enc_seq = x_swin_enc.flatten(2).transpose(1, 2)
        x_swin = x_swin_hr + x_swin_enc_seq
        for blk in self.swin_decoder:
            x_swin = blk(x_swin)
        x_swin_dec = x_swin.transpose(1, 2).view(B, C, H, W)  # (B,256,32,32)

        # 6. 特征融合与上采样
        x_fuse = x_swin_dec + x_lr_32  # (B,256,32,32)
        x_fuse = self.combine_block1(x_fuse)  # (B,256,32,32)

        # 第一次上采样：32x32 -> 64x64, channels 256 -> 64 (PixelShuffle(2))
        x_up1 = self.upsample1(x_fuse)  # (B,64,64,64)

        # 与 lr_path 的 64x64 特征融合（更合理的 skip）
        x_fuse2 = x_up1 + x_lr_64  # (B,64,64,64)
        x_fuse2 = self.combine_block2(x_fuse2)  # (B,64,64,64)

        # 第二次上采样：16x16 -> 256x256, channels 64 -> 1 (PixelShuffle(8))
        x_up2 = self.upsample2(x_fuse2)  # (B,1,256,256)

        # 最终卷积：通道 1 -> 1
        out = self.final_conv(x_up2)  # (B,1,512,512)
        return out
