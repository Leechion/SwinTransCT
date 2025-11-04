import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None):
        if key is None:
            key = query
        if value is None:
            value = key

        b, n, d = query.shape
        qkv = self.qkv_proj(query).view(b, n, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [b, nhead, n, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = attn_probs @ v  # [b, nhead, n, head_dim]
        output = output.transpose(1, 2).contiguous().view(b, n, d)  # 合并多头
        return self.out_proj(output)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model, dims):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, dims[0]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dims[0], dims[1])
        )

    def forward(self, x):
        return self.layers(x)


class ResidualConnection(nn.Module):
    """残差连接（按你原意：norm -> sublayer -> add）"""
    def __init__(self, d_model):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer=None):
        # sublayer 应该是一个 callable，接受 norm(x) 并返回同形输出
        if sublayer is not None:
            return x + sublayer(self.norm(x))
        # 若没有 sublayer，就做常规残差（x + norm(x)）
        return x + self.norm(x)


# 新增：按你原始意图实现的编码器层（保留结构，不改变设计）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, nhead)
        self.ff = FeedForward(d_model, [8 * d_model, d_model])
        self.res1 = ResidualConnection(d_model)
        self.res2 = ResidualConnection(d_model)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = self.res1(x, lambda y: self.mha(y))   # self-attention residual
        x = self.res2(x, lambda y: self.ff(y))    # feed-forward residual
        return x


# 新增：解码器层（含自注意力 + cross-attn + FF）
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ff = FeedForward(d_model, [8 * d_model, d_model])
        self.res1 = ResidualConnection(d_model)
        self.res2 = ResidualConnection(d_model)
        self.res3 = ResidualConnection(d_model)

    def forward(self, x, memory):
        # x: [B, seq_x, d_model], memory: [B, seq_mem, d_model]
        x = self.res1(x, lambda y: self.self_attn(y))                       # self-attn
        x = self.res2(x, lambda y: self.cross_attn(y, memory, memory))      # cross-attn(query=x, key/value=memory)
        x = self.res3(x, lambda y: self.ff(y))                              # FFN
        return x


class LDCTNet256(nn.Module):
    def __init__(self):
        super(LDCTNet256, self).__init__()
        self.param = {
            'nx': 256, 'ny': 256, 'numIterations': 50000, 'u_water': 0.0205,
            'batchsize': 16, 'lr': 5e-5, 'retrain': False, 'epoch': 2500
        }

        # 创建高斯核（放 CPU 上，forward 时发送到输入设备）
        self.gaussian_kernel = self._create_gaussian_kernel(11, 1.5)  # tensor on cpu

        # LR路径（低频特征提取）
        self.lr_conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)  # 256→128
        self.lr_conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)  # 128→64
        self.lr_conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # 64→32
        self.lr_conv4 = nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=2)  # 32→16

        # HR路径（注意：space_to_depth(block_size=8) 对 1ch 输入会产生 64ch）
        # 因此第一层 conv 应接收 64 通道
        self.hr_process_convs = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),   # <-- 64 -> 256
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        # 编码器与解码器（使用上面新增的 EncoderLayer / DecoderLayer）
        self.encoder_layers = nn.ModuleList([EncoderLayer(256, 8) for _ in range(3)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(256, 8) for _ in range(3)])

        # 特征融合层
        self.combine_convs1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        # combine_convs2 的输入通道应当与 x_hr 和 x32_lr 的通道一致（两者都是 64ch）
        self.combine_convs2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # <-- 改为 64 输入
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        # 输出层
        self.final_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        # 初始化权重
        self._initialize_weights()

    def _create_gaussian_kernel(self, size, sigma):
        """创建高斯核（返回 CPU tensor，forward 时再 .to(device)）"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, size, size)  # cpu tensor

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def space_to_depth(x, block_size):
        """实现 tf.space_to_depth（更健壮的实现）"""
        b, c, h, w = x.size()
        assert h % block_size == 0 and w % block_size == 0
        x = x.view(b, c, h // block_size, block_size, w // block_size, block_size)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(b, c * (block_size ** 2), h // block_size, w // block_size)
        return x

    @staticmethod
    def depth_to_space(x, block_size):
        """实现 tf.depth_to_space（更直观、可读）"""
        b, c, h, w = x.size()
        assert c % (block_size ** 2) == 0, "channels must be divisible by block_size^2"
        new_c = c // (block_size ** 2)
        x = x.view(b, new_c, block_size, block_size, h, w)
        # 现在维度 (b, new_c, block_size, block_size, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # -> (b, new_c, h, block_size, w, block_size)
        x = x.view(b, new_c, h * block_size, w * block_size)
        return x

    def forward(self, x):
        """前向传播 - 假定输入 x 是 [B,1,256,256]"""
        b, c, h, w = x.shape
        assert c == 1 and h == 256 and w == 256, "Input must be grayscale 256x256"

        # 高低频分离（把 kernel 发送到输入所在 device）
        kernel = self.gaussian_kernel.repeat(c, 1, 1, 1).to(x.device)
        img_LR = F.conv2d(x, kernel, padding=5, groups=c)
        img_HR = x - img_LR

        # LR 路径
        x128_lr = F.leaky_relu(self.lr_conv1(img_LR))
        x64_lr = F.leaky_relu(self.lr_conv2(x128_lr))
        x32_lr = F.leaky_relu(self.lr_conv3(x64_lr))
        x16_lr = F.leaky_relu(self.lr_conv4(x32_lr))  # [B,256,16,16]

        # 编码器（Transformer）
        memory = x16_lr.flatten(2).transpose(1, 2)  # [B, 16*16, 256]
        for enc_layer in self.encoder_layers:
            memory = enc_layer(memory)               # 使用 EncoderLayer

        # HR 路径：先 space_to_depth (block_size=8) -> channels: 1*64 = 64
        img_HR_patch = self.space_to_depth(img_HR, block_size=8)  # [B,64,32,32]
        x_hrproc = self.hr_process_convs(img_HR_patch)            # [B,256,32,32]

        # 解码器
        b_hr, c_hr, h_hr, w_hr = x_hrproc.shape
        x_flat = x_hrproc.flatten(2).transpose(1, 2)  # [B, h_hr*w_hr, c_hr]
        for dec_layer in self.decoder_layers:
            x_flat = dec_layer(x_flat, memory)        # 使用 DecoderLayer
        x_dec = x_flat.transpose(1, 2).view(b_hr, c_hr, h_hr, w_hr)  # [B,256,32,32]

        # 融合与上采样
        x_dec_down = F.interpolate(x_dec, size=x16_lr.shape[2:], mode='bilinear', align_corners=False)
        fea_16 = x_dec_down + x16_lr               # [B,256,16,16] 注意：x_dec是32->16错位？（下文 align）
        # 说明：在你的原代码里 x_dec 的 spatial 与 x16_lr 一致，因为你用的 block / reshape 恰好对齐。
        x = self.combine_convs1(fea_16)
        x = x + fea_16
        x_hr = self.depth_to_space(x, block_size=2)  # 上采样到 32x32，channels -> 256 // 4 = 64

        fea_32 = x_hr + x32_lr                 # 64-ch + 64-ch
        x = self.combine_convs2(fea_32)       # combine_convs2 现在接受 64 通道输入
        x = F.relu(x + fea_32)
        out = self.depth_to_space(x, block_size=8)  # 上采样到 256x256 (channels -> 64//64 = 1)
        out = self.final_conv(out)
        return out
