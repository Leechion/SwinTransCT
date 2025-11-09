import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================
# 基础模块定义
# ==========================

class FeedForward(nn.Module):
    """前馈网络层"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    """Robust MHA: supports different token lengths for q and kv (N_q != N_kv)."""
    def __init__(self, dim, num_heads=8, debug=False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        # separate projections for clarity
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.debug = debug

    def forward(self, x_q, x_kv=None):
        """
        x_q: (B, N_q, C)
        x_kv: (B, N_kv, C)  (if None, x_kv = x_q)
        returns: (B, N_q, C)
        """
        if x_kv is None:
            x_kv = x_q

        # basic checks
        if x_q.dim() != 3 or x_kv.dim() != 3:
            raise RuntimeError(f"MHA expects 3D inputs (B, N, C). Got {x_q.shape} and {x_kv.shape}")

        B, N_q, Cq = x_q.shape
        B2, N_kv, Ck = x_kv.shape
        if B != B2:
            raise RuntimeError("Batch size mismatch between q and kv")
        if Cq != self.dim or Ck != self.dim:
            raise RuntimeError(f"Last dim must equal declared dim={self.dim}. Got Cq={Cq}, Ck={Ck}")

        # projections
        q = self.q_proj(x_q)        # (B, N_q, dim)
        k = self.k_proj(x_kv)       # (B, N_kv, dim)
        v = self.v_proj(x_kv)       # (B, N_kv, dim)

        # reshape into heads
        # q -> (B, heads, N_q, head_dim)
        q = q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        # k, v -> (B, heads, N_kv, head_dim)
        k = k.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        if self.debug:
            print(f"MHA q shape {q.shape}, k shape {k.shape}, v shape {v.shape}")

        # scaled dot-product attention
        # attn: (B, heads, N_q, N_kv)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)

        # out: (B, heads, N_q, head_dim)
        out = attn @ v

        # merge heads -> (B, N_q, dim)
        out = out.transpose(1, 2).reshape(B, N_q, self.dim)
        out = self.out_proj(out)
        return out



class EncoderLayer(nn.Module):
    """Transformer Encoder层"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, 8 * dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder层"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        self.ffn = FeedForward(dim, 8 * dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, memory):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), memory)
        x = x + self.ffn(self.norm3(x))
        return x


# ==========================
# 主模型定义
# ==========================

class LDCTNet256(nn.Module):
    def __init__(self):
        super().__init__()

        # Gaussian blur conv kernel (可以用固定卷积核或nn.Conv2d)
        self.gaussian = nn.Conv2d(1, 1, kernel_size=11, padding=5, bias=False)
        self.init_gaussian()

        # LR分支卷积层
        self.lr_conv1 = nn.Conv2d(1, 16, 5, 2, 2)
        self.lr_conv2 = nn.Conv2d(16, 32, 5, 2, 2)
        self.lr_conv3_lr = nn.Conv2d(32, 64, 5, 2, 2)
        self.lr_conv4_lr = nn.Conv2d(64, 256, 5, 2, 2)
        self.lr_conv3_hr = nn.Conv2d(32, 64, 5, 2, 2)
        self.lr_conv4_hr = nn.Conv2d(64, 128, 5, 2, 2)
        self.lr_conv_final = nn.Conv2d(128, 256, 5, 2, 2)

        # Transformer编码层
        self.encoder = nn.ModuleList([EncoderLayer(256, 8) for _ in range(3)])

        # HR分支卷积层
        self.hr_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Transformer解码层
        self.decoder = nn.ModuleList([DecoderLayer(256, 8) for _ in range(3)])

        # Combine部分
        self.combine_32 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.combine_64 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def init_gaussian(self, sigma=1.5):
        """初始化高斯卷积核"""
        import numpy as np
        ksize = 11
        ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.gaussian.weight.copy_(kernel)

    def forward(self, x):
        # Gaussian blur
        img_LR = self.gaussian(x)
        img_HR = x - img_LR

        # LR branch
        x1 = F.leaky_relu(self.lr_conv1(img_LR))
        x2 = F.leaky_relu(self.lr_conv2(x1))
        x3_lr = F.leaky_relu(self.lr_conv3_lr(x2))
        x4_lr = F.leaky_relu(self.lr_conv4_lr(x3_lr))
        x3_hr = F.leaky_relu(self.lr_conv3_hr(x2))
        x4_hr = F.leaky_relu(self.lr_conv4_hr(x3_hr))
        x_lr_final = F.leaky_relu(self.lr_conv_final(x4_hr))

        # Transformer Encoder
        b, c, h, w = x_lr_final.shape
        memory = x_lr_final.view(b, c, -1).permute(0, 2, 1)
        for layer in self.encoder:
            memory = layer(memory)

        # HR branch
        x_hr = F.pixel_unshuffle(img_HR, downscale_factor=16)
        x_hr = self.hr_conv(x_hr)
        b, c, h, w = x_hr.shape
        x_hr = x_hr.view(b, c, -1).permute(0, 2, 1)
        for layer in self.decoder:
            x_hr = layer(x_hr, memory)
        x_hr = x_hr.permute(0, 2, 1).view(b, c, h, w)

        # Combine
        x = x_hr + x4_lr
        x = self.combine_32(x) + x_hr
        x = F.pixel_shuffle(x, upscale_factor=2)

        x = x + x3_lr
        x = self.combine_64(x) + x
        out = F.pixel_shuffle(x, upscale_factor=8)

        return out
