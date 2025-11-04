import torch
import torch.nn.functional as F
from torch import nn
import math


def calculate_psnr(img1, img2, data_range=1.0):
    """
    计算 PSNR（峰值信噪比）
    :param img1: 预测图像张量 (B, C, H, W)
    :param img2: 真实图像张量 (B, C, H, W)
    :param data_range: 图像数据范围（默认1.0，需与预处理后的图像范围匹配）
    :return: 批量图像的平均 PSNR
    """
    # 计算 MSE（均方误差）
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')  # MSE为0时PSNR无穷大
    # PSNR公式：PSNR = 10 * log10((data_range^2) / MSE)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr.item()


def calculate_ssim(img1, img2, data_range=1.0, window_size=11, window_sigma=1.5):
    """
    计算 SSIM（结构相似性指数）
    参考 PyTorch 官方实现思路，适配批量图像
    :param img1: 预测图像张量 (B, C, H, W)
    :param img2: 真实图像张量 (B, C, H, W)
    :param data_range: 图像数据范围（默认1.0）
    :param window_size: 高斯窗口大小（默认11，需为奇数）
    :param window_sigma: 高斯窗口标准差（默认1.5）
    :return: 批量图像的平均 SSIM
    """
    device = img1.device
    C1 = (0.01 * data_range) ** 2  # 稳定性常数1
    C2 = (0.03 * data_range) ** 2  # 稳定性常数2

    # 生成1D高斯窗口
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * window_sigma ** 2)) 
                          for x in range(window_size)]).to(device)
    gauss = gauss / gauss.sum()  # 归一化

    # 扩展为2D窗口（1, 1, window_size, window_size）
    window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(img1.shape[1], 1, window_size, window_size)

    # 计算图像均值（高斯滤波）
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

    # 计算图像方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    # SSIM公式
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()  # 批量平均


def calculate_rmse(img1, img2):
    """
    计算 RMSE（均方根误差）
    :param img1: 预测图像张量 (B, C, H, W)
    :param img2: 真实图像张量 (B, C, H, W)
    :return: 批量图像的平均 RMSE
    """
    # RMSE公式：RMSE = sqrt(MSE)
    mse = F.mse_loss(img1, img2, reduction='mean')
    rmse = torch.sqrt(mse)
    return rmse.item()


class TrainingRecorder:
    """
    训练过程记录器：将每个epoch的训练/验证指标保存到CSV文件
    支持断点续训时自动续写，避免数据丢失
    """
    def __init__(self, save_path, args=None):
        """
        Args:
            save_path: CSV文件保存路径（如"logs/training_metrics.csv"）
            args: 训练参数（可选，用于记录训练配置到CSV头部）
        """
        self.save_path = save_path
        self._init_csv(args)

    def _init_csv(self, args):
        """初始化CSV文件：若不存在则创建并写入表头，若存在则跳过表头"""
        # 创建父目录（若不存在）
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # 检查文件是否已存在
        file_exists = os.path.isfile(self.save_path)
        
        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_psnr", "train_ssim", "train_rmse",
                "val_loss", "val_psnr", "val_ssim", "val_rmse", "lr", "timestamp"
            ])
            
            # 若文件新创建，先写入训练配置和表头
            if not file_exists:
                # 写入训练参数配置（便于后续复现实验）
                if args is not None:
                    f.write("# 训练配置参数\n")
                    for key, value in vars(args).items():
                        f.write(f"# {key}: {value}\n")
                    f.write("# \n")  # 空行分隔配置和数据
                
                # 写入指标表头
                writer.writeheader()

    def record_epoch(self, epoch, train_metrics, val_metrics, current_lr):
        """
        记录单个epoch的指标到CSV文件
        Args:
            epoch: 当前epoch编号（int）
            train_metrics: 训练集指标字典（来自train_one_epoch返回值）
            val_metrics: 验证集指标字典（来自validate返回值）
            current_lr: 当前学习率（float）
        """
        # 构建指标字典（与CSV表头对应）
        metrics_dict = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_psnr": round(train_metrics["psnr"], 4),
            "train_ssim": round(train_metrics["ssim"], 6),
            "train_rmse": round(train_metrics["rmse"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_psnr": round(val_metrics["psnr"], 4),
            "val_ssim": round(val_metrics["ssim"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "lr": round(current_lr, 8),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 追加写入CSV
        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            writer.writerow(metrics_dict)
        
        # 打印记录提示（可选）
        print(f"[指标记录] Epoch {epoch} 指标已保存到 {self.save_path}")






# 封装指标计算类（方便训练中调用）
class ImageMetrics(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        """
        一次性计算 PSNR、SSIM、RMSE
        :param pred: 预测图像 (B, C, H, W)
        :param target: 真实图像 (B, C, H, W)
        :return: metrics_dict（包含三个指标的数值）
        """
        psnr = calculate_psnr(pred, target, self.data_range)
        ssim = calculate_ssim(pred, target, self.data_range)
        rmse = calculate_rmse(pred, target)
        return {
            'psnr': psnr,
            'ssim': ssim,
            'rmse': rmse
        }