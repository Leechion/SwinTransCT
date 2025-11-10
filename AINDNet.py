import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# --------------------------
# 工具函数（对齐原始逻辑）
# --------------------------
def down_sample(x, scale_factor_h, scale_factor_w):
    """下采样：对齐tf.image.resize_bilinear（PyTorch用interpolate实现）"""
    return F.interpolate(
        x, 
        scale_factor=(1/scale_factor_h, 1/scale_factor_w), 
        mode='bilinear', 
        align_corners=False
    )

class ResBlock(nn.Module):
    """基础残差块（对齐原始resBlock）"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, scale=1.0):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        # 卷积层（无激活函数，原始逻辑）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.scale = scale
        # 初始化（对齐tf.truncated_normal/stddev=0.02）
        init.trunc_normal_(self.conv1.weight, std=0.02)
        init.trunc_normal_(self.conv2.weight, std=0.02)
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x
        tmp = F.relu(self.conv1(x))
        tmp = self.conv2(tmp)
        tmp *= self.scale
        return residual + tmp

class AIN(nn.Module):
    """自适应实例归一化（对齐原始ain函数）"""
    def __init__(self, in_channels, hidden_channels=64):
        super(AIN, self).__init__()
        self.param_free_norm = nn.InstanceNorm2d(in_channels, affine=False, eps=1e-5)
        # 噪声图处理卷积层（对齐原始tf.layers.conv2d）
        self.conv1 = nn.Conv2d(3, hidden_channels, kernel_size=5, padding=2)  # 输入噪声图是3通道
        self.conv_gamma = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1)
        # 初始化（对齐tf.contrib.layers.variance_scaling_initializer(0.02)）
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv_gamma.weight, mode='fan_out', nonlinearity='linear')
        init.kaiming_normal_(self.conv_beta.weight, mode='fan_out', nonlinearity='linear')
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv_gamma.bias)
        init.zeros_(self.conv_beta.bias)

    def forward(self, noise_map, x_init):
        # 噪声图下采样到x_init尺寸（对齐tf.image.resize_bilinear）
        noise_map_down = F.interpolate(
            noise_map, 
            size=x_init.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        # 参数无关归一化（对齐param_free_norm）
        x = self.param_free_norm(x_init)
        # 噪声图特征提取
        tmp = F.relu(self.conv1(noise_map_down))
        gamma = self.conv_gamma(tmp)
        beta = self.conv_beta(tmp)
        # 自适应调整
        x = x * (1 + gamma) + beta
        return x

class AINResBlock(nn.Module):
    """AIN残差块（对齐原始ain_resblock）"""
    def __init__(self, in_channels, hidden_channels=64):
        super(AINResBlock, self).__init__()
        self.ain1 = AIN(in_channels, hidden_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.ain2 = AIN(in_channels, hidden_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 初始化
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='leaky_relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='leaky_relu')
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)

    def forward(self, noise_map, x_init):
        x = self.ain1(noise_map, x_init)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = self.conv1(x)
        
        x = self.ain2(noise_map, x)
        x = F.leaky_relu(x, negative_slope=0.02)
        x = self.conv2(x)
        
        return x + x_init

class FCN_Avg(nn.Module):
    """噪声预测分支（对齐原始FCN_Avg/FCN_Avgp，两者结构完全一致）"""
    def __init__(self):
        super(FCN_Avg, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init.zeros_(m.bias)

    def forward(self, x):
        # 下采样分支（down_noise_map）
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        x1 = F.avg_pool2d(x1, kernel_size=4, stride=4, padding=0)  # 对齐tf.layers.average_pooling2d
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.avg_pool2d(x1, kernel_size=2, stride=2, padding=0)
        down_noise_map = F.relu(self.conv5(x1))
        
        # 上采样分支（noise_map）
        image_shape = x.shape[2:]  # (H, W)
        y = F.interpolate(down_noise_map, size=image_shape, mode='bilinear', align_corners=False)
        y = F.relu(self.conv6(y))
        noise_map = F.relu(self.conv7(y))
        
        return down_noise_map, noise_map

class ResUpsampleAndSum(nn.Module):
    """上采样+跳跃连接（对齐原始res_upsample_and_sum）"""
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(ResUpsampleAndSum, self).__init__()
        self.res_block = ResBlock(out_channels, out_channels)  # 处理编码器特征x2
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, stride=kernel_size, padding=0
        )
        # 初始化转置卷积（对齐tf.truncated_normal/stddev=0.02）
        init.trunc_normal_(self.deconv.weight, std=0.02)
        init.zeros_(self.deconv.bias)

    def forward(self, x1, x2):
        """
        x1: 解码器输入（需上采样）
        x2: 编码器对应层级特征（需残差处理后融合）
        """
        x2 = self.res_block(x2)
        deconv = self.deconv(x1)
        # 确保deconv与x2尺寸一致（应对整除问题）
        if deconv.shape != x2.shape:
            deconv = F.interpolate(deconv, size=x2.shape[2:], mode='bilinear', align_corners=False)
        return deconv + x2

class AINDNetRecon(nn.Module):
    """核心重建网络（U-Net结构，对齐原始AINDNet_recon）"""
    def __init__(self):
        super(AINDNetRecon, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.ain_res1_1 = AINResBlock(64)
        self.ain_res1_2 = AINResBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ain_res2_1 = AINResBlock(128)
        self.ain_res2_2 = AINResBlock(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.ain_res3_1 = AINResBlock(256)
        self.ain_res3_2 = AINResBlock(256)
        self.ain_res3_3 = AINResBlock(256)
        self.ain_res3_4 = AINResBlock(256)
        self.ain_res3_5 = AINResBlock(256)
        
        # 解码器
        self.up4 = ResUpsampleAndSum(256, 128)  # in_channels=256, out_channels=128
        self.ain_res4_1 = AINResBlock(128)
        self.ain_res4_2 = AINResBlock(128)
        self.ain_res4_3 = AINResBlock(128)
        
        self.up5 = ResUpsampleAndSum(128, 64)   # in_channels=128, out_channels=64
        self.ain_res5_1 = AINResBlock(64)
        self.ain_res5_2 = AINResBlock(64)
        
        # 输出层
        self.conv6 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        # 初始化
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        init.trunc_normal_(self.conv6.weight, std=0.02)
        init.zeros_(self.conv1.bias)
        init.zeros_(self.conv2.bias)
        init.zeros_(self.conv3.bias)
        init.zeros_(self.conv6.bias)

    def forward(self, x, noise_map):
        # 编码器
        conv1 = F.relu(self.conv1(x))
        conv1 = self.ain_res1_1(noise_map, conv1)
        conv1 = self.ain_res1_2(noise_map, conv1)
        pool1 = F.avg_pool2d(conv1, kernel_size=2, stride=2, padding=1)  # 对齐slim.avg_pool2d(SAME)
        
        conv2 = F.relu(self.conv2(pool1))
        conv2 = self.ain_res2_1(noise_map, conv2)
        conv2 = self.ain_res2_2(noise_map, conv2)
        pool2 = F.avg_pool2d(conv2, kernel_size=2, stride=2, padding=1)
        
        conv3 = F.relu(self.conv3(pool2))
        conv3 = self.ain_res3_1(noise_map, conv3)
        conv3 = self.ain_res3_2(noise_map, conv3)
        conv3 = self.ain_res3_3(noise_map, conv3)
        conv3 = self.ain_res3_4(noise_map, conv3)
        conv3 = self.ain_res3_5(noise_map, conv3)
        
        # 解码器
        up4 = self.up4(conv3, conv2)
        up4 = self.ain_res4_1(noise_map, up4)
        up4 = self.ain_res4_2(noise_map, up4)
        up4 = self.ain_res4_3(noise_map, up4)
        
        up5 = self.up5(up4, conv1)
        up5 = self.ain_res5_1(noise_map, up5)
        up5 = self.ain_res5_2(noise_map, up5)
        
        out = self.conv6(up5)
        return out

class AINDNet(nn.Module):
    """整体网络（对齐原始AINDNet）"""
    def __init__(self):
        super(AINDNet, self).__init__()
        self.fcn_avg = FCN_Avg()
        self.recon_net = AINDNetRecon()

    def forward(self, x):
        """
        输入：x（3通道含噪图像，如低剂量CT图像，shape: [B, 3, H, W]）
        输出：noise_map（预测的噪声图）, out（重建后的图像）
        """
        # 噪声预测分支
        down_noise_map, noise_map = self.fcn_avg(x)
        # 噪声图融合（0.8*上采样下采样噪声图 + 0.2*原始噪声图）
        upsample_noise_map = F.interpolate(
            down_noise_map, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        noise_map = 0.8 * upsample_noise_map + 0.2 * noise_map
        # 重建分支（残差连接：out = recon + input）
        recon_out = self.recon_net(x, noise_map)
        out = recon_out + x
        return noise_map, out



