import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===============================
# SSIM计算（单通道 256×256）
# ===============================
def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """计算单通道 SSIM"""
    def gaussian_window(window_size, sigma=1.5):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(1)
        return g @ g.t()
    
    window = gaussian_window(window_size).to(img1.device)
    window = window.expand(1, 1, window_size, window_size)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# ===============================
# HybridLoss（SSIM + MSE + 感知）
# ===============================
class HybridLoss(nn.Module):
    """
    适配医学 CT 图像的混合损失:
    - 0.4 * SSIM Loss
    - 0.4 * MSE Loss
    - 0.2 * Perceptual (ResNet18) Loss
    """
    def __init__(self, w_ssim=0.4, w_mse=0.4, w_perc=0.2, feature_layer=6):
        super().__init__()
        self.w_ssim = w_ssim
        self.w_mse = w_mse
        self.w_perc = w_perc

        # 加载ResNet18作为医学感知网络（可换成Med3D）
        resnet = models.resnet18(pretrained=True)
        layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        ][:feature_layer]  # 只保留浅层特征
        self.feature_extractor = nn.Sequential(*layers)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        """
        pred, target: [B, 1, 256, 256]  灰度 CT 图像（归一化到0-1）
        """
        # 1️⃣ SSIM loss
        ssim_loss = 1 - ssim(pred, target)

        # 2️⃣ MSE loss
        mse_loss = F.mse_loss(pred, target)

        # 3️⃣ 感知特征损失（使用ResNet18前几层）
        # ResNet 需要3通道输入，灰度图扩展为3通道
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        with torch.no_grad():
            f_pred = self.feature_extractor(pred_rgb)
            f_target = self.feature_extractor(target_rgb)

        perceptual_loss = F.l1_loss(f_pred, f_target)

        # 4️⃣ 混合加权
        total_loss = (self.w_ssim * ssim_loss +
                      self.w_mse * mse_loss +
                      self.w_perc * perceptual_loss)

        return total_loss
