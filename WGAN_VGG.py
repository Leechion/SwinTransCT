import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np

# -----------------------------
# Generator
# -----------------------------
class WGAN_VGG_generator(nn.Module):
    def __init__(self):
        super(WGAN_VGG_generator, self).__init__()
        layers = [nn.Conv2d(1,32,3,1,1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32,32,3,1,1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32,1,3,1,1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Discriminator: PatchGAN
# -----------------------------
class WGAN_VGG_Discriminator_FC_to_Conv(nn.Module):
    """
    PatchGAN 判别器，支持任意尺寸输入
    """
    def __init__(self, in_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)  # 输出 patch map
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Feature extractor (VGG19)
# -----------------------------
class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # VGG固定权重

    def forward(self, x):
        return self.feature_extractor(x)

# -----------------------------
# WGAN-VGG main class
# -----------------------------
class WGAN_VGG(nn.Module):
    def __init__(self):
        super(WGAN_VGG, self).__init__()
        self.generator = WGAN_VGG_generator()
        self.discriminator = WGAN_VGG_Discriminator_FC_to_Conv(in_channels=1)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()

    # -------------------------
    # Discriminator loss
    # -------------------------
    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    # -------------------------
    # Generator loss
    # -------------------------
    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    # -------------------------
    # Perceptual loss
    # -------------------------
    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)

        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)

        # 自动对齐 feature map
        if fake_feature.shape[2:] != real_feature.shape[2:]:
            real_feature = F.interpolate(real_feature, size=fake_feature.shape[2:], mode='bilinear', align_corners=False)

        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    # -------------------------
    # Gradient penalty
    # -------------------------
    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.rand(y.size(0), 1, 1, 1, device=y.device)
        interp = (a*y + (1-a)*fake).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.ones(d_interp.size(), device=y.device)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * lambda_
        return gradient_penalty

# -----------------------------
# Test example
# -----------------------------
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WGAN_VGG().to(device)
    x = torch.randn(1, 1, 256, 256).to(device)
    y = torch.randn(1, 1, 256, 256).to(device)

    out = model.generator(x)
    print("Generator output shape:", out.shape)  # [1,1,256,256]

    loss = model.p_loss(x, y)
    print("Perceptual loss:", loss.item())

    d_loss = model.d_loss(x, y)
    print("Discriminator loss:", d_loss.item())
