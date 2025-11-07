import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import argparse

# 导入你训练代码中的自定义模块（保持路径一致）
from dataset import CTDataset, get_pair_list  # 复用数据集类，避免重复编码
from model_improve import LDCTNet_Swin_improve  # 你的模型
from utils import ImageMetrics  # 复用指标计算模块（确保与训练时一致）

# --------------------------
# 工具函数：图像对比保存（简化版，适配测试集）
# --------------------------
def save_test_comparison(ld_img, nd_img, output_img, save_path, img_name):
    """保存测试集图像对比图（LD输入 → ND标签 → 模型输出）"""
    os.makedirs(save_path, exist_ok=True)
    
    # 归一化到[0,1]（适配可视化）
    def normalize(img_tensor):
        img = img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()  # (B,1,H,W) → (H,W)
        img_min, img_max = img.min(), img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)
    
    # 归一化三张图
    ld_norm = normalize(ld_img)
    nd_norm = normalize(nd_img)
    output_norm = normalize(output_img)
    
    # 横向拼接（LD→ND→输出）
    comparison = np.hstack([ld_norm, nd_norm, output_norm])  # (H, 3*W)
    comparison = (comparison * 255).astype(np.uint8)  # 转为8位灰度图
    
    # 保存文件
    save_filename = f"test_{img_name}_comparison.png"
    save_filepath = os.path.join(save_path, save_filename)
    Image.fromarray(comparison, mode='L').save(save_filepath)
    return save_filepath

# --------------------------
# 核心测试函数
# --------------------------
def test_model(args):
    # 1. 设备配置（与训练一致）
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"=" * 60)
    print(f"测试配置：")
    print(f"  设备: {device}")
    print(f"  测试集根目录: {args.data_dir}")
    print(f"  模型路径: {args.model_path}")
    print(f"  结果保存目录: {args.result_dir}")
    print(f"  批量大小: {args.batch_size}")
    print(f"=" * 60)

    # 2. 加载测试集（复用训练时的Dataset，确保预处理一致）
    print("\n[1/4] 加载测试集...")
    test_pairs = get_pair_list(args.data_dir, split="test")  # 读取测试集配对列表
    test_dataset = CTDataset(test_pairs, target_size=256)  # 与训练时保持相同target_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 测试集不打乱，便于对应文件名
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"  测试集样本数：{len(test_dataset)} 张")
    print(f"  测试集批次：{len(test_loader)} 批")
    print(f"  图像尺寸：{test_dataset[0][0].shape}")

    # 3. 初始化模型与指标计算器
    print("\n[2/4] 初始化模型与工具...")
    # 初始化模型（与训练时一致）
    model = LDCTNet_Swin_improve().to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载训练好的模型权重（关键！）
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        # 兼容不同保存格式（仅加载模型权重，忽略优化器等）
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)  # 若直接保存了model.state_dict()
        print(f"  成功加载模型：{args.model_path}")
    else:
        raise FileNotFoundError(f"  错误：未找到模型文件 {args.model_path}")

    # 指标计算器（data_range与训练时一致，假设为[0,1]）
    metrics_fn = ImageMetrics(data_range=1.0).to(device)
    model.eval()  # 切换为评估模式（禁用Dropout、BatchNorm冻结）

    # 4. 创建结果保存目录
    os.makedirs(args.result_dir, exist_ok=True)
    csv_save_path = os.path.join(args.result_dir, "test_metrics.csv")  # 指标CSV
    img_save_path = os.path.join(args.result_dir, "test_comparisons")  # 对比图

    # 5. 批量测试与指标计算
    print("\n[3/4] 开始批量测试...")
    # 初始化结果列表（存储每张图的指标）
    test_results = []
    total_psnr = 0.0
    total_ssim = 0.0
    total_rmse = 0.0

    # 进度条
    progress_bar = tqdm(test_loader, desc="[Test] Batch Processing", unit="batch")
    for batch_idx, (ld_imgs, nd_imgs) in enumerate(progress_bar):
        # 数据移至设备
        ld_imgs = ld_imgs.to(device, non_blocking=True)
        nd_imgs = nd_imgs.to(device, non_blocking=True)
        batch_size = ld_imgs.size(0)

        # 关闭梯度计算（加速+省内存）
        with torch.no_grad():
            outputs = model(ld_imgs)  # 模型推理

        # 计算当前批次所有图像的指标（按样本逐个计算，确保与文件名对应）
        for idx in range(batch_size):
            # 提取单张图像（B,1,H,W）→（1,1,H,W）
            ld_img = ld_imgs[idx:idx+1]
            nd_img = nd_imgs[idx:idx+1]
            output_img = outputs[idx:idx+1]

            # 计算指标（复用训练时的metrics_fn，确保一致性）
            metrics = metrics_fn(output_img, nd_img)
            psnr = metrics["psnr"]
            ssim = metrics["ssim"]
            rmse = metrics["rmse"]

            # 累计总指标（用于计算平均值）
            total_psnr += psnr
            total_ssim += ssim
            total_rmse += rmse

            # 获取当前图像的文件名（关键：与Dataset的配对列表顺序一致）
            img_filename = test_pairs[batch_idx * args.batch_size + idx][0].split("/")[-1]  # 取LD图像的文件名
            img_name = img_filename.split(".")[0]  # 去掉后缀（如"0359"）

            # 保存单张图像结果
            test_results.append({
                "filename": img_filename,  # 图像文件名（如"0359.png"）
                "psnr": round(psnr, 4),    # 保留4位小数
                "ssim": round(ssim, 4),
                "rmse": round(rmse, 6),
                "comparison_path": ""  # 对比图路径（可选）
            })

            # 可选：每N张保存一张对比图（避免保存过多）
            if (batch_idx * args.batch_size + idx) % args.save_img_freq == 0:
                comp_path = save_test_comparison(ld_img, nd_img, output_img, img_save_path, img_name)
                test_results[-1]["comparison_path"] = comp_path  # 更新对比图路径

        # 进度条更新
        progress_bar.set_postfix({
            "batch_avg_psnr": f"{total_psnr/( (batch_idx+1)*batch_size ):.2f}",
            "batch_avg_ssim": f"{total_ssim/( (batch_idx+1)*batch_size ):.4f}"
        })

    # 6. 计算测试集整体指标（均值±标准差）
    avg_psnr = total_psnr / len(test_dataset)
    avg_ssim = total_ssim / len(test_dataset)
    avg_rmse = total_rmse / len(test_dataset)

    # 计算指标标准差（用于统计稳定性）
    psnr_list = [r["psnr"] for r in test_results]
    ssim_list = [r["ssim"] for r in test_results]
    rmse_list = [r["rmse"] for r in test_results]
    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_rmse = np.std(rmse_list)

    # 7. 保存结果到CSV
    print("\n[4/4] 保存测试结果...")
    # 转换为DataFrame并保存
    results_df = pd.DataFrame(test_results)
    # 在CSV最后添加整体统计行
    stats_row = pd.DataFrame({
        "filename": ["average±std"],
        "psnr": [f"{avg_psnr:.4f}±{std_psnr:.4f}"],
        "ssim": [f"{avg_ssim:.4f}±{std_ssim:.4f}"],
        "rmse": [f"{avg_rmse:.6f}±{std_rmse:.6f}"],
        "comparison_path": ["-"]
    })
    results_df = pd.concat([results_df, stats_row], ignore_index=True)
    results_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"  指标CSV已保存：{csv_save_path}")

    # 8. 打印测试总结
    print("\n" + "=" * 80)
    print("测试集整体性能：")
    print(f"  PSNR：{avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"  SSIM：{avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  RMSE：{avg_rmse:.6f} ± {std_rmse:.6f}")
    print(f"=" * 80)
    if args.save_img_freq > 0:
        print(f"  对比图保存目录：{img_save_path}（每{args.save_img_freq}张保存1张）")

# --------------------------
# 命令行参数配置（与训练代码风格一致）
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据与模型路径
    parser.add_argument("--data_dir", type=str, default="./ND_LD_Paired_Data", 
                        help="数据集根目录（含train/val/test子文件夹）")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.pth", 
                        help="训练好的模型路径（best_model.pth）")
    parser.add_argument("--result_dir", type=str, default="./test_results", 
                        help="测试结果保存目录（CSV+对比图）")
    # 测试配置
    parser.add_argument("--batch_size", type=int, default=8, help="测试批量大小（根据GPU内存调整）")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号（-1表示CPU）")
    parser.add_argument("--save_img_freq", type=int, default=10, 
                        help="每N张保存一张对比图（0表示不保存）")
    args = parser.parse_args()

    # 执行测试
    test_model(args)