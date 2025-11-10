import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FBPReconstructor(nn.Module):
    """修复后的 FBP 重建模块（向量化 Ram-Lak 频域滤波 + 正确的 grid_sample 使用）"""
    def __init__(self, img_size=256, num_angles=180, det_count=256, device=None):
        super(FBPReconstructor, self).__init__()
        self.img_size = img_size
        self.num_angles = num_angles
        self.det_count = det_count
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # 投影到角度的模拟卷积（你原来用 proj_conv 从图像生成投影特征）
        # 输出通道数 = num_angles, 保持 spatial = (H,W) 与输入一致
        self.proj_conv = nn.Conv2d(1, self.num_angles, kernel_size=3, padding=1, bias=False)
        nn.init.xavier_normal_(self.proj_conv.weight)

        # Ram-Lak 频率响应（real, length = det_count/2+1 for rfft），用 register_buffer 管理 device
        freq = np.fft.rfftfreq(self.det_count, d=1.0 / self.det_count)  # shape (det_count//2+1,)
        ramlak = np.abs(freq).astype(np.float32)  # Ram-Lak magnitude
        self.register_buffer('ramlak_freq', torch.from_numpy(ramlak))  # shape (det_count//2+1,)

    def forward(self, x):
        """
        x: [B,1,H,W]  (H,W expected = img_size)
        returns: fbp_img [B,1,img_size,img_size]
        """
        B = x.shape[0]
        device = self.device
        x = x.to(device)

        # 1) 生成模拟投影特征并将空间维度（H）求和以得到投影：proj_feat [B, num_angles, H, W]
        proj_feat = self.proj_conv(x)  # [B, num_angles, H, W]
        # 假设探测器维度在最后一个 dim (W)，把中间维 H 汇总（即模拟环绕整圈的积分）
        proj_sim = torch.sum(proj_feat, dim=2)  # [B, num_angles, det_count]  (det_count == W expected)

        # 2) 在探测器维度上做 Ram-Lak 频域滤波（向量化，批次和角度并行）
        # 使用 rfft/irfft：proj_sim -> fft -> multiply freq response -> ifft
        proj_fft = torch.fft.rfft(proj_sim, dim=-1)  # shape [B, num_angles, det_count//2+1], complex
        # ramlak_freq shape [det_count//2+1], cast to complex and broadcast multiply
        ramlak = self.ramlak_freq.to(proj_fft.dtype).to(device)  # real tensor
        proj_fft_filtered = proj_fft * ramlak.unsqueeze(0).unsqueeze(0)  # broadcast multiply
        filtered_proj = torch.fft.irfft(proj_fft_filtered, n=self.det_count, dim=-1)  # [B, num_angles, det_count]

        # 3) 准备反投影坐标 grid（统一用 img_size 尺寸）
        # 生成坐标网格 xx, yy ∈ [-1, 1]
        x_lin = torch.linspace(-1.0, 1.0, steps=self.img_size, device=device)
        y_lin = torch.linspace(-1.0, 1.0, steps=self.img_size, device=device)
        yy, xx = torch.meshgrid(y_lin, x_lin, indexing='ij')  # yy: (H,W), xx: (H,W)
        # 注意：在你的推导里 t = x*cos + y*sin（x,y在[-1,1]），所以使用 xx,yy 即可
        angles = torch.linspace(0.0, np.pi, steps=self.num_angles, device=device)

        # 4) 逐角度反投影（向量化展开部分维度以兼容 grid_sample）
        # 预分配输出
        fbp_img = torch.zeros(B, 1, self.img_size, self.img_size, device=device)

        # 我们将为每个角度构建相同 shape 的 grid: [H, W, 2], 然后扩展为 [B, H, W, 2]
        # grid_sample expects source [N, C, H_src, W_src] and grid [N, H_out, W_out, 2] where
        # grid[...,0] is x coord (width), grid[...,1] is y coord (height), both in [-1,1].
        # Our source for sampling will be proj_i expanded to [B,1,1,det_count] -> expanded to [B,1,H_out,det_count]
        # so source H_src=1, W_src=det_count.

        for i in range(self.num_angles):
            theta = angles[i]
            # t = x*cos + y*sin  (xx,yy ∈ [-1,1])
            t = xx * torch.cos(theta) + yy * torch.sin(theta)  # shape (H,W), in [-1,1]
            # det coord normalized in [-1,1] for width dimension
            det_coords_x = t  # already in [-1,1]

            # build grid for sampling from source with H_src=1 and W_src=det_count
            # grid's last dim: (x_coord_for_width, y_coord_for_height)
            # since source H_src == 1, y must be 0 (center)
            grid = torch.stack([det_coords_x, torch.zeros_like(det_coords_x)], dim=-1)  # (H,W,2)
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B,H,W,2)

            # extract filtered projection for angle i: [B, det_count] -> reshape [B,1,1,det_count]
            proj_i = filtered_proj[:, i, :].unsqueeze(1).unsqueeze(2)  # [B,1,1,det_count]
            # expand to [B,1,H_out,det_count] so grid_sample can map (H_out,W_out) -> (H_src,W_src)
            proj_i_exp = proj_i.expand(-1, -1, self.img_size, -1)  # [B,1,H_out,det_count]

            # sample: src [B, C, H_src=H_out, W_src=det_count] ??? careful:
            # proj_i_exp has shape [B,1,H_out,det_count], grid is [B,H_out,W_out,2], OK.
            # Use align_corners=True to align linear mapping consistently
            proj_interp = F.grid_sample(proj_i_exp, grid, mode='bilinear', padding_mode='border', align_corners=True)
            # proj_interp shape: [B,1,H_out,W_out]
            fbp_img = fbp_img + proj_interp

        # normalize by number of angles (and multiply by pi as in classical FBP scaling)
        fbp_img = fbp_img / float(self.num_angles) * float(np.pi)
        return fbp_img


# 其他模块（CNNRefiner、FBPNet_256PNG）保持结构但 move to same device in forward
class CNNRefiner(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=16):
        super(CNNRefiner, self).__init__()
        self.down1 = self._conv_block(in_channels, num_filters)
        self.down2 = self._conv_block(num_filters, num_filters*2)
        self.down3 = self._conv_block(num_filters*2, num_filters*4)
        self.bottleneck = self._conv_block(num_filters*4, num_filters*8)
        self.up3 = self._up_block(num_filters*8, num_filters*4)
        self.up2 = self._up_block(num_filters*4, num_filters*2)
        self.up1 = self._up_block(num_filters*2, num_filters)
        self.out_conv = nn.Conv2d(num_filters, out_channels, kernel_size=1, padding=0)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0),
            self._conv_block(out_ch, out_ch)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        b = self.bottleneck(F.max_pool2d(d3, 2))
        u3 = self.up3(b) + d3
        u2 = self.up2(u3) + d2
        u1 = self.up1(u2) + d1
        refined = self.out_conv(u1)
        return refined

class FBPNet(nn.Module):
    def __init__(self, device=None):
        super(FBPNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.fbp_reconstructor = FBPReconstructor(img_size=256, device=self.device)
        self.cnn_refiner = CNNRefiner()
        # move modules to device
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        fbp_img = self.fbp_reconstructor(x)
        refined_img = self.cnn_refiner(fbp_img)
        final = refined_img + fbp_img
        return final



