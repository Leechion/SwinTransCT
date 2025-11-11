import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
import argparse
import numpy as np
import random
from torchvision.utils import make_grid

# 导入loss模块（确保路径正确，若报错需调整导入路径）
from loss import HybridLoss
from dataset2 import CTDataset, get_pair_list  # 我的数据集类
from model_improve import LDCTNet_Swin_improve  # 我的LDCTNet_Swin模型
from Trans_model_writer import LDCTNet256  # 你的LDCTNet_Swin模型
from Red_CNN import RED_CNN  # 加载Red_CNN
from model import LDCTNet_Swin
########################################################################################
from utils import ImageMetrics  # 指标计算模块（PSNR/SSIM/RMSE）
from utils import TrainingRecorder  # 指标计算模块（PSNR/SSIM/RMSE）

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def save_image_grid(ld_img, nd_img, output_img, save_path, epoch):
    """
    布局：低剂量输入 → 正常剂量标签 → 模型输出
    Args:
        ld_img: 低剂量CT输入 [B,1,H,W]
        nd_img: 正常剂量CT标签 [B,1,H,W]
        output_img: 模型输出 [B,1,H,W]
        save_path: 图像保存目录
        epoch: 当前epoch编号
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 图像归一化到[0,1]（按单个样本归一化，保证亮度均衡）
    def normalize(img):
        img = img.detach().cpu()
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)  # 避免除零

    # 只取第一个样本（[B,1,H,W] → [1,H,W]），避免多样本混乱
    ld_sample = ld_img[0:1]  # 取第一个样本，保持[1,1,H,W]格式
    nd_sample = nd_img[0:1]
    output_sample = output_img[0:1]

    # 归一化
    ld_norm = normalize(ld_sample)
    nd_norm = normalize(nd_sample)
    output_norm = normalize(output_sample)

    # 拼接为一行3列：低剂量（左）→ 正常剂量（中）→ 模型输出（右）
    # 按水平方向（dim=3）拼接，最终形状：[1,1,H, 3*W]
    comparison_img = torch.cat([ld_norm, nd_norm, output_norm], dim=3)

    # 转换为PIL图像（适配保存）
    from PIL import Image
    # 处理维度：[1,1,H,3W] → [H,3W]（去掉批量和通道维度）
    img_np = comparison_img.squeeze(0).squeeze(0).numpy()
    img_np = (img_np * 255).astype(np.uint8)  # [0,1] → [0,255]

    # 保存文件（文件名含epoch，便于按顺序查看）
    save_filename = f"epoch_{epoch:03d}_comparison.png"  # 03d补零（如005、010）
    save_filepath = os.path.join(save_path, save_filename)
    Image.fromarray(img_np, mode='L').save(save_filepath)  # mode='L'：灰度图格式

    # 打印保存日志
    print(f"[图像保存] 一行三列对比图已保存：{save_filepath}")
    print(f"[图像布局] 左：低剂量输入 | 中：正常剂量标签 | 右：模型输出")


def train_one_epoch(model, train_loader, criterion, optimizer, metrics_fn, device, epoch, writer):
    """训练一个epoch，同步计算损失和图像质量指标"""
    model.train()
    # 初始化累计变量（按样本数加权）
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_samples = 0

    # 进度条显示训练过程
    progress_bar = tqdm(train_loader, desc=f"[Train] Epoch {epoch:3d}", unit="batch")
    for batch_idx, (ld_img, nd_img) in enumerate(progress_bar):
        # 数据移至设备（CPU/GPU）
        ld_img = ld_img.to(device, non_blocking=True)  # 低剂量CT（输入）
        nd_img = nd_img.to(device, non_blocking=True)  # 正常剂量CT（标签）
        batch_size = ld_img.size(0)
        total_samples += batch_size

        # 1. 清零梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(ld_img)

        # 3. 计算损失和指标
        loss = criterion(outputs, nd_img)
        metrics = metrics_fn(outputs, nd_img)  # 一次性获取PSNR/SSIM/RMSE

        # 4. 反向传播与参数更新
        loss.backward()
        optimizer.step()  # 更新权重

        # 5. 累计损失和指标
        total_loss += loss.item() * batch_size
        total_psnr += metrics["psnr"] * batch_size
        total_ssim += metrics["ssim"] * batch_size
        total_rmse += metrics["rmse"] * batch_size

        # 6. 实时更新进度条（显示当前batch的损失和PSNR）
        progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.6f}",
            "batch_psnr": f"{metrics['psnr']:.2f}",
            "batch_ssim": f"{metrics['ssim']:.4f}"
        })

        # 7. 记录batch级日志（每10个batch写一次TensorBoard）
        if (batch_idx + 1) % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_PSNR", metrics["psnr"], global_step)
            writer.add_scalar("Train/Batch_SSIM", metrics["ssim"], global_step)
            writer.add_scalar("Train/Batch_RMSE", metrics["rmse"], global_step)

    # 8. 计算epoch级平均指标
    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_rmse = total_rmse / total_samples

    # 9. 记录epoch级日志（写入TensorBoard）
    writer.add_scalar("Train/Epoch_Loss", avg_loss, epoch)
    writer.add_scalar("Train/Epoch_PSNR", avg_psnr, epoch)
    writer.add_scalar("Train/Epoch_SSIM", avg_ssim, epoch)
    writer.add_scalar("Train/Epoch_RMSE", avg_rmse, epoch)

    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "rmse": avg_rmse
    }


def validate(model, val_loader, criterion, metrics_fn, device, epoch, writer):
    """验证模型，计算损失和指标，同步记录图像对比"""
    model.eval()
    # 初始化累计变量
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0
    total_samples = 0

    # 关闭梯度计算（加速验证，避免内存占用）
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch:3d}", unit="batch")
        for batch_idx, (ld_img, nd_img) in enumerate(progress_bar):
            ld_img = ld_img.to(device, non_blocking=True)
            nd_img = nd_img.to(device, non_blocking=True)
            batch_size = ld_img.size(0)
            total_samples += batch_size

            # 1. 前向传播
            outputs = model(ld_img)

            # 2. 计算损失和指标
            loss = criterion(outputs, nd_img)
            metrics = metrics_fn(outputs, nd_img)

            # 3. 累计损失和指标
            total_loss += loss.item() * batch_size
            total_psnr += metrics["psnr"] * batch_size
            total_ssim += metrics["ssim"] * batch_size
            total_rmse += metrics["rmse"] * batch_size

            # 4. 实时更新进度条
            progress_bar.set_postfix({
                "batch_loss": f"{loss.item():.6f}",
                "batch_psnr": f"{metrics['psnr']:.2f}",
                "batch_ssim": f"{metrics['ssim']:.4f}"
            })

            # 5. 记录验证集图像对比（每5个epoch，仅取第一个batch的第一张图）
            if epoch % 10 == 0 and batch_idx == 0:
                # 图像归一化到[0,1]（TensorBoard显示需要）
                def normalize_img(img_tensor):
                    img = img_tensor.cpu().numpy()[0, 0]  # (B,1,H,W) → (H,W)
                    img_min, img_max = img.min(), img.max()
                    return (img - img_min) / (img_max - img_min + 1e-8)  # 避免除零

                # 提取并归一化3张图：低剂量输入、正常剂量标签、模型输出
                ld_img_norm = normalize_img(ld_img)
                nd_img_norm = normalize_img(nd_img)
                output_norm = normalize_img(outputs)

                # 写入TensorBoard（三张图横向排列，便于对比）
                writer.add_image(f"Val/Epoch_{epoch}_LowDose", ld_img_norm, epoch, dataformats="HW")
                writer.add_image(f"Val/Epoch_{epoch}_FullDose", nd_img_norm, epoch, dataformats="HW")
                writer.add_image(f"Val/Epoch_{epoch}_ModelOutput", output_norm, epoch, dataformats="HW")

    # 6. 计算验证集epoch级平均指标
    avg_loss = total_loss / total_samples
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    avg_rmse = total_rmse / total_samples

    # 7. 记录验证集epoch级日志
    writer.add_scalar("Val/Epoch_Loss", avg_loss, epoch)
    writer.add_scalar("Val/Epoch_PSNR", avg_psnr, epoch)
    writer.add_scalar("Val/Epoch_SSIM", avg_ssim, epoch)
    writer.add_scalar("Val/Epoch_RMSE", avg_rmse, epoch)

    return {
        "loss": avg_loss,
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "rmse": avg_rmse
    }


def main(args):
    # 1. 设备配置（优先GPU，无GPU则用CPU）
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"=" * 60)
    print(f"训练配置：")
    print(f"  设备: {device}")
    print(f"  数据集根目录: {args.data_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  初始学习率: {args.lr}")
    print(f"  日志保存目录: {args.log_dir}")
    print(f"  模型保存目录: {args.save_dir}")
    print(f"=" * 60)

    # 2. 加载数据集（适配256×256图像）
    print("\n[1/5] 加载数据集...")
    # 生成训练/验证集图像对列表
    train_dataset, val_dataset = get_pair_list(
        data_dir=args.data_dir,
        target_size=256
    )

    # 2. 创建DataLoader（批量加载，训练时shuffle=True）
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,  # 根据CPU核心数调整
        pin_memory=True  # 加速GPU训练
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"  训练集：{len(train_dataset)} 样本 | {len(train_loader)} 批次")
    print(f"  验证集：{len(val_dataset)} 样本 | {len(val_loader)} 批次")
    print(f"  图像尺寸：{train_dataset[0][0].shape}（已Resize到256×256）")

    # 3. 初始化模型、损失函数、优化器、指标计算器
    print("\n[2/5] 初始化模型与工具...")

    ########################################################################################################
    # 初始化模型（TransCT模型）
    #model = LDCTNet256().to(device)

    # 初始化模型（Red_CNN模型）
    # model = RED_CNN().to(device)

    #初始化LDCTNet_Swin（输入尺寸256×256，与数据集匹配）
    # model = LDCTNet_Swin(input_size=(256, 256), base_channels=16,swin_window_size=7,swin_num_heads=8 ).to(device)
    model = LDCTNet_Swin_improve().to(device)
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"模型结构:")
    print(model)
    #############################################################################################################

    # 损失函数（MSE适合CT剂量恢复，可后续替换为MSE+SSIM混合损失）
    criterion = HybridLoss().to(device)  
    #criterion = nn.MSELoss().to(device)

    # 优化器（Adam + 权重衰减防过拟合）
    optimizer = optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=args.lr,
        weight_decay=1e-5  # 权重衰减系数，可根据需求调整
    )
    # 学习率调度器（验证损失不下降时降低学习率）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,   # 余弦周期长度（一般设为总epoch数）
        eta_min=1e-6         # 最低学习率，防止完全归零

    )
    # 指标计算器（data_range需与预处理后图像范围匹配，这里假设[0,1]）
    metrics_fn = ImageMetrics(data_range=1.0).to(device)

    # 4. 日志与模型保存配置
    print("\n[3/5] 配置日志与模型保存...")
    # 创建日志和模型保存目录（不存在则自动创建）
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # 新增：初始化训练记录器（保存到log_dir下的metrics.csv）
    recorder = TrainingRecorder(
        save_path=os.path.join(args.log_dir, "training_metrics.csv"),
        args=args  # 传入训练参数，记录到CSV头部
    )

    # 最佳模型记录（用验证集ssim作为评判标准，ssim越高模型越好）
    best_val_ssim = 0.0   
    best_val_psnr = 0.0
    best_val_epoch = 0

    # 5. 加载预训练模型（可选，支持断点续训）
    if args.resume != "":
        print(f"\n[4/5] 加载预训练模型：{args.resume}")
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            # 加载模型权重和优化器状态
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # 恢复训练进度和最佳指标
            start_epoch = checkpoint["epoch"] + 1
            best_val_psnr = checkpoint["best_val_psnr"]
            best_val_epoch = checkpoint["best_val_epoch"]
            print(f"  恢复训练：从Epoch {start_epoch}开始")
            print(f"  历史最佳：Epoch {best_val_epoch} | Val PSNR: {best_val_psnr:.2f}")
        else:
            print(f"  警告：未找到预训练模型文件，将从头开始训练")
            start_epoch = 0
    else:
        start_epoch = 0

    # 6. 训练主循环
    print("\n[5/5] 开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, metrics_fn, device, epoch, writer
        )
        # 验证一个epoch
        val_metrics = validate(
            model, val_loader, criterion, metrics_fn, device, epoch, writer
        )
        # 调整学习率（基于验证集损失）
        scheduler.step()

        # 新增：记录当前epoch的学习率（用于CSV保存）
        current_lr = optimizer.param_groups[0]["lr"]
        # --------------------------
        # 新增：本地保存图像对比网格
        # --------------------------
        save_freq = 5  # 每5个epoch保存一次（可调整为1、10等）
        # 获取第一个batch（用于保存图像对比网格）
        val_first_batch = next(iter(val_loader))
        if epoch % save_freq == 0:
            # 获取第一个batch的第一个样本
            ld_sample, nd_sample = val_first_batch
            ld_sample = ld_sample.to(device, non_blocking=True)
            nd_sample = nd_sample.to(device, non_blocking=True)

            # 模型推理（仅对第一个样本）
            model.eval()
            with torch.no_grad():
                output_sample = model(ld_sample)

            # 调用保存函数（保存路径：log_dir/images）
            save_image_grid(
                ld_img=ld_sample,
                nd_img=nd_sample,
                output_img=output_sample,
                save_path=os.path.join(args.image_dir, "images"),  # 保存目录
                epoch=epoch
            )
        # --------------------------
        # 新增代码结束
        # --------------------------

        # 7. 控制台打印epoch总结（格式化输出，清晰易读）
        print(f"\n" + "=" * 80)
        print(f"Epoch {epoch:3d}/{args.epochs - 1:3d} | 训练集指标：")
        print(f"  损失：{train_metrics['loss']:.6f} | PSNR：{train_metrics['psnr']:.2f} dB")
        print(f"  SSIM：{train_metrics['ssim']:.4f} | RMSE：{train_metrics['rmse']:.6f}")
        print(f"Epoch {epoch:3d}/{args.epochs - 1:3d} | 验证集指标：")
        print(f"  损失：{val_metrics['loss']:.6f} | PSNR：{val_metrics['psnr']:.2f} dB")
        print(f"  SSIM：{val_metrics['ssim']:.4f} | RMSE：{val_metrics['rmse']:.6f}")
        print(f"=" * 80)

        # 新增：调用recorder记录指标到CSV
        recorder.record_epoch(epoch, train_metrics, val_metrics, current_lr)

        # 8. 保存最佳模型（验证集PSNR更高则更新）
        if val_metrics["ssim"] > best_val_ssim:
            best_val_ssim = val_metrics["ssim"]
            best_val_epoch = epoch
            # 保存模型权重、优化器状态、训练进度
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_psnr": best_val_psnr,
                "best_val_epoch": best_val_epoch,
                "val_metrics": val_metrics  # 保存当前验证集指标
            }
            best_model_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"[保存最佳模型] Epoch {epoch} | 验证集PSNR：{best_val_psnr:.2f} dB | 验证集SSIM：{best_val_ssim:.4f}")

        # 9. 保存定期 checkpoint（每50个epoch保存一次，便于回溯）
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics
            }
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"[保存定期 checkpoint] Epoch {epoch} | 路径：{checkpoint_path}")

    # 10. 训练结束：打印总结并关闭writer
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"最佳模型：Epoch {best_val_epoch} | 验证集PSNR：{best_val_psnr:.2f} dB｜验证集：{best_val_ssim:.4f}")
    print(f"最佳模型路径：{os.path.join(args.save_dir, 'best_model.pth')}")
    writer.close()
    print("TensorBoard 日志已保存在：{}".format(args.log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./ND_LD_Paired_Data_0.5")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--image_dir", type=str, default="./image_loader")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()
    main(args)


    


