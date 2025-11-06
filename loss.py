import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    Hybrid Loss for CT reconstruction:
    L = 0.7*L1 + 0.2*(1-SSIM) + 0.1*EdgeLoss
    输入输出尺寸: [B, 1, H, W]
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1,
                 window_size=11, window_sigma=1.5, device='cpu'):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.window_size = window_size
        self.device = device

        # 高斯卷积核，用于 SSIM
        self.gaussian_kernel = self.create_gaussian_kernel(window_size, window_sigma)

        # Sobel 卷积核，用于 Edge Loss
        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device).view(1,1,3,3)

    def create_gaussian_kernel(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size//2
        g = torch.exp(-(coords**2)/(2*sigma**2))
        g = g / g.sum()
        kernel_2d = g[:,None] @ g[None,:]  # 外积得到 2D
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        return kernel_2d

    def ssim(self, img1, img2, C1=1e-4, C2=9e-4):
        """计算 SSIM，输入 [B,1,H,W]"""
        mu1 = F.conv2d(img1, self.gaussian_kernel, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.gaussian_kernel, padding=self.window_size//2)
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self.gaussian_kernel, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.gaussian_kernel, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.gaussian_kernel, padding=self.window_size//2) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq + C1)*(sigma1_sq+sigma2_sq + C2))
        return ssim_map.mean()

    def edge_loss(self, pred, target):
        """基于 Sobel 边缘的 L1 损失"""
        gx_pred = F.conv2d(pred, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred, self.sobel_y, padding=1)
        gx_target = F.conv2d(target, self.sobel_x, padding=1)
        gy_target = F.conv2d(target, self.sobel_y, padding=1)
        loss = F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
        return loss

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_loss = 1 - self.ssim(pred, target)
        edge = self.edge_loss(pred, target)
        loss = self.alpha * l1 + self.beta * ssim_loss + self.gamma * edge
        return loss
