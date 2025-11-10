import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import argparse
from matplotlib.gridspec import GridSpec

# 导入你训练代码中的自定义模块（保持路径一致）
from dataset import CTDataset, get_pair_list  # 复用数据集类
from model_improve import LDCTNet_Swin_improve  # 你的模型
from Red_CNN import RED_CNN  # 你的模型
from Trans_model_writer import LDCTNet256  # 你的模型
from utils import ImageMetrics  # 复用指标计算模块

# --------------------------
# 工具函数：图像归一化与组图绘制（核心修复）
# --------------------------
def normalize_img(img_tensor):
    """将张量图像归一化到[0,1]（适配可视化）"""
    img = img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()  # (B,1,H,W) → (H,W)
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)  # 避免除零

def create_best_worst_group_plot(selected_samples, save_path, dpi=600):
    """
    生成高清组图：每行对应一个样本（LD → ND → Output），共9行（3最高PSNR+3最高SSIM+3最低RMSE）
    修复：直接使用样本的 psnr/ssim/rmse 字段，无需依赖 metrics 字典
    """
    # SCI风格配置（保持图表专业美观）
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'figure.facecolor': 'white'
    })

    # 创建9行3列的组图（9个样本×3张图）
    fig = plt.figure(figsize=(15, 30), dpi=dpi)
    gs = GridSpec(9, 3, figure=fig, hspace=0.3, wspace=0.1)  # 紧凑布局

    # 定义子图标题（每个样本的3张图标题）
    subplot_titles = ['Low-Dose CT (Input)', 'Normal-Dose CT (GT)', 'Model Output']
    # 定义样本类型标签（区分不同筛选条件）
    sample_labels = [
        'Top 1 PSNR', 'Top 2 PSNR', 'Top 3 PSNR',
        'Top 1 SSIM', 'Top 2 SSIM', 'Top 3 SSIM',
        'Bottom 1 RMSE', 'Bottom 2 RMSE', 'Bottom 3 RMSE'
    ]

    # 逐个绘制样本（核心修复：读取 psnr/ssim/rmse 字段）
    for idx, (sample, label) in enumerate(zip(selected_samples, sample_labels)):
        ld_img = normalize_img(sample['ld_img'])
        nd_img = normalize_img(sample['nd_img'])
        output_img = normalize_img(sample['output_img'])
        # 直接从样本中读取指标（不再依赖 metrics 字典）
        psnr_val = sample['psnr']
        ssim_val = sample['ssim']
        rmse_val = sample['rmse']
        filename = sample['filename']

        # 绘制LD图（第1列）
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(ld_img, cmap='gray', aspect='equal')
        ax1.set_title(subplot_titles[0], fontsize=11, fontweight='bold')
        ax1.axis('off')  # 隐藏坐标轴（组图更简洁）

        # 绘制ND图（第2列）
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(nd_img, cmap='gray', aspect='equal')
        ax2.set_title(subplot_titles[1], fontsize=11, fontweight='bold')
        ax2.axis('off')

        # 绘制Output图（第3列）
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.imshow(output_img, cmap='gray', aspect='equal')
        ax3.set_title(subplot_titles[2], fontsize=11, fontweight='bold')
        ax3.axis('off')

        # 在每行左侧添加样本信息（文件名+指标）→ 直接使用读取的字段
        info_text = f"{label}\nFile: {filename}\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}\nRMSE: {rmse_val:.6f}"
        fig.text(0.01, 0.97 - idx/9, info_text, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

    # 总标题
    fig.suptitle('Selected Test Samples: Best Performance (PSNR/SSIM) & Lowest RMSE',
                 fontsize=16, fontweight='bold', y=0.995)

    # 保存高清组图（支持PNG/EPS）
    plt.tight_layout(rect=[0.05, 0.01, 1.0, 0.99])  # 预留左侧信息栏空间
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  高清组图已保存：{save_path}")

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
    print(f"  图像保存目录: {args.img_save_dir}")
    print(f"  批量大小: {args.batch_size}")
    print(f"=" * 60)

    # 2. 创建保存目录（自动创建不存在的目录）
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    csv_save_path = os.path.join(args.result_dir, "test_metrics.csv")  # 指标CSV
    group_img_save_path = os.path.join(args.img_save_dir, "best_worst_samples_group.png")  # 组图

    # 3. 加载测试集（复用训练时的Dataset，确保预处理一致）
    print("\n[1/5] 加载测试集...")
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

    # 4. 初始化模型与指标计算器
    print("\n[2/5] 初始化模型与工具...")
    model = RED_CNN().to(device)
    print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载模型权重
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, weights_only=False, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"  成功加载模型：{args.model_path}")
    else:
        raise FileNotFoundError(f"  错误：未找到模型文件 {args.model_path}")

    metrics_fn = ImageMetrics(data_range=1.0).to(device)
    model.eval()  # 切换为评估模式

    # 5. 批量测试与指标记录（核心：移除 metrics 字段，避免冗余）
    print("\n[3/5] 开始批量测试...")
    test_results = []  # 存储每张图的结果（含指标+图像张量）
    progress_bar = tqdm(test_loader, desc="[Test] Batch Processing", unit="batch")

    for batch_idx, (ld_imgs, nd_imgs) in enumerate(progress_bar):
        ld_imgs = ld_imgs.to(device, non_blocking=True)
        nd_imgs = nd_imgs.to(device, non_blocking=True)
        batch_size = ld_imgs.size(0)

        with torch.no_grad():
            outputs = model(ld_imgs)  # 模型推理

        # 逐张图像计算指标并保存信息
        for idx in range(batch_size):
            # 提取单张图像（保留原始张量，用于后续绘图）
            ld_img = ld_imgs[idx:idx+1].clone()  # 克隆避免张量被覆盖
            nd_img = nd_imgs[idx:idx+1].clone()
            output_img = outputs[idx:idx+1].clone()

            # 计算指标（已修复：无 .item()）
            metrics = metrics_fn(output_img, nd_img)
            psnr = metrics["psnr"]  # 原生float
            ssim = metrics["ssim"]
            rmse = metrics["rmse"]

            # 获取文件名
            ld_filepath = test_pairs[batch_idx * args.batch_size + idx][0]
            filename = os.path.basename(ld_filepath)  # 如"0359.png"

            # 保存当前图像的完整结果（移除 metrics 字段，避免KeyError）
            test_results.append({
                "filename": filename,
                "psnr": round(psnr, 4),
                "ssim": round(ssim, 4),
                "rmse": round(rmse, 6),
                "ld_img": ld_img,
                "nd_img": nd_img,
                "output_img": output_img
            })

        # 进度条更新（显示当前平均指标）
        current_avg = {
            "psnr": np.mean([r["psnr"] for r in test_results]),
            "ssim": np.mean([r["ssim"] for r in test_results]),
            "rmse": np.mean([r["rmse"] for r in test_results])
        }
        progress_bar.set_postfix({
            "avg_psnr": f"{current_avg['psnr']:.2f}",
            "avg_ssim": f"{current_avg['ssim']:.4f}",
            "avg_rmse": f"{current_avg['rmse']:.6f}"
        })

    # 6. 保存完整指标到CSV（含每张图+平均值）
    print("\n[4/5] 保存测试指标CSV...")
    # 转换为DataFrame（仅保留数值列，排除张量）
    csv_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ld_img", "nd_img", "output_img"]} 
                          for r in test_results])
    
    # 计算整体平均值并添加到CSV最后一行
    avg_row = {
        "filename": "average",
        "psnr": round(csv_df["psnr"].mean(), 4),
        "ssim": round(csv_df["ssim"].mean(), 4),
        "rmse": round(csv_df["rmse"].mean(), 6)
    }
    csv_df = pd.concat([csv_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存CSV（支持中文文件名，utf-8-sig编码）
    csv_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"  完整指标CSV已保存：{csv_save_path}")
    print(f"  CSV包含 {len(csv_df)-1} 张测试图的指标 + 1行平均值")

    # 7. 筛选最优/最差样本并生成组图（无变化）
    print("\n[5/5] 筛选样本并生成高清组图...")
    # 筛选规则：去重（避免同一指标筛选到同一样本），取Top3/Bottom3
    # 1) 最高3个PSNR（降序排列）
    top3_psnr = sorted(test_results, key=lambda x: x["psnr"], reverse=True)[:3]
    # 2) 最高3个SSIM（排除已选的PSNR样本）
    remaining_for_ssim = [r for r in test_results if r not in top3_psnr]
    top3_ssim = sorted(remaining_for_ssim, key=lambda x: x["ssim"], reverse=True)[:3]
    # 3) 最低3个RMSE（排除已选的PSNR/SSIM样本）
    remaining_for_rmse = [r for r in test_results if r not in top3_psnr + top3_ssim]
    bottom3_rmse = sorted(remaining_for_rmse, key=lambda x: x["rmse"])[:3]

    # 组合所有选中的样本（共9个）
    selected_samples = top3_psnr + top3_ssim + bottom3_rmse

    # 生成并保存高清组图
    create_best_worst_group_plot(selected_samples, group_img_save_path, dpi=args.dpi)

    # 打印筛选结果总结
    print("\n筛选样本总结：")
    print("=" * 60)
    for i, sample in enumerate(top3_psnr, 1):
        print(f"Top {i} PSNR: {sample['filename']} | PSNR: {sample['psnr']:.2f} dB")
    for i, sample in enumerate(top3_ssim, 1):
        print(f"Top {i} SSIM: {sample['filename']} | SSIM: {sample['ssim']:.4f}")
    for i, sample in enumerate(bottom3_rmse, 1):
        print(f"Bottom {i} RMSE: {sample['filename']} | RMSE: {sample['rmse']:.6f}")
    print("=" * 60)

# --------------------------
# 命令行参数配置
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--data_dir", type=str, default="./ND_LD_Paired_Data_0.5", 
                        help="数据集根目录（含train/val/test子文件夹）")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model_Red_Cnn.pth", 
                        help="训练好的模型路径")
    parser.add_argument("--result_dir", type=str, default="./test_results", 
                        help="测试指标CSV保存目录")
    parser.add_argument("--img_save_dir", type=str, default="./test_image_floder", 
                        help="测试组图保存目录（自动创建）")
    # 测试配置
    parser.add_argument("--batch_size", type=int, default=1, help="测试批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号（-1表示CPU）")
    parser.add_argument("--dpi", type=int, default=600, help="组图分辨率（默认600，越高越清晰）")
    args = parser.parse_args()

    # 执行测试
    test_model(args)