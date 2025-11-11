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
from matplotlib.colors import LinearSegmentedColormap

# -------------------------- 新增：HDF5数据集支持（如果需要用HDF5测试集，取消注释） --------------------------
# import h5py
# import torchvision.transforms as T
# class CTHDF5Dataset(torch.utils.data.Dataset):
#     def __init__(self, hdf5_path, target_size=256):
#         self.hdf5_path = hdf5_path
#         self.target_size = target_size
#         with h5py.File(hdf5_path, "r") as f:
#             self.num_samples = f.attrs["num_images"]
#         self.transform = T.Compose([T.Resize((target_size, target_size), T.InterpolationMode.BILINEAR)])
#     def __len__(self):
#         return self.num_samples
#     def __getitem__(self, idx):
#         with h5py.File(self.hdf5_path, "r") as f:
#             ld_img = f["LD"][idx].astype(np.float32)
#             nd_img = f["ND"][idx].astype(np.float32)
#         ld_tensor = torch.from_numpy(ld_img).unsqueeze(0).float()
#         nd_tensor = torch.from_numpy(nd_img).unsqueeze(0).float()
#         ld_tensor = self.transform(ld_tensor)
#         nd_tensor = self.transform(nd_tensor)
#         return ld_tensor, nd_tensor

# -------------------------- 导入你训练代码中的自定义模块 --------------------------
from dataset import CTDataset, get_pair_list  # 默认为PNG数据集，如需HDF5请替换上面的类
from model_improve import LDCTNet_Swin_improve  # 你的模型
from Red_CNN import RED_CNN
from Trans_model_writer import LDCTNet256
from utils import ImageMetrics  # 复用指标计算模块

# -------------------------- 核心工具函数 --------------------------
def normalize_img(img_tensor):
    """将张量图像归一化到[0,1]（适配可视化）"""
    img = img_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()  # (B,1,H,W) → (H,W)
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)  # 避免除零

def compute_diff_heatmap(output_img, nd_img):
    """计算Output与ND的差异热力图（归一化到[0,1]）"""
    output_norm = normalize_img(output_img)
    nd_norm = normalize_img(nd_img)
    diff = np.abs(output_norm - nd_norm)
    # 归一化差异图（确保热力图颜色分布均匀）
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    return diff_norm

def create_selected_group_plot(selected_samples, save_path, dpi=600):
    """
    生成最终组图：2行（SSIM最好2张 + PSNR最好2张）×4列（ND、LD、Output、差异热力图）
    """
    # SCI风格配置（Times New Roman字体，专业美观）
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.unicode_minus': False,
        'axes.linewidth': 0.8,
        'figure.facecolor': 'white',
        'savefig.bbox': 'tight'
    })

    # 定义配色（热力图用viridis，专业且区分度高）
    cmap_heat = "viridis"

    # 创建2行4列的组图（2个类别×2张图 = 4行？不：2个类别，每个类别2张图 → 共4行）
    # 修正布局：4行（Top1 SSIM + Top2 SSIM + Top1 PSNR + Top2 PSNR）×4列
    fig = plt.figure(figsize=(20, 20), dpi=dpi)
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.2)

    # 子图列标题
    col_titles = ['Normal-Dose CT (ND)', 'Low-Dose CT (LD)', 'Model Output', 'Difference Heatmap (|Output - ND|)']
    # 样本行标签
    sample_labels = [
        'Top 1 SSIM', 'Top 2 SSIM',
        'Top 1 PSNR', 'Top 2 PSNR'
    ]

    # 逐个绘制样本（4行×4列）
    for row_idx, (sample, label) in enumerate(zip(selected_samples, sample_labels)):
        # 归一化图像（适配可视化）
        nd_img = normalize_img(sample['nd_img'])
        ld_img = normalize_img(sample['ld_img'])
        output_img = normalize_img(sample['output_img'])
        diff_heatmap = compute_diff_heatmap(sample['output_img'], sample['nd_img'])

        # 获取指标
        psnr_val = sample['psnr']
        ssim_val = sample['ssim']
        filename = sample['filename']

        # -------------------------- 第1列：ND（高剂量CT） --------------------------
        ax1 = fig.add_subplot(gs[row_idx, 0])
        im1 = ax1.imshow(nd_img, cmap='gray', aspect='equal')
        if row_idx == 0:  # 第一行添加列标题
            ax1.set_title(col_titles[0], fontsize=14, fontweight='bold', pad=20)
        ax1.axis('off')

        # -------------------------- 第2列：LD（低剂量CT） --------------------------
        ax2 = fig.add_subplot(gs[row_idx, 1])
        im2 = ax2.imshow(ld_img, cmap='gray', aspect='equal')
        if row_idx == 0:
            ax2.set_title(col_titles[1], fontsize=14, fontweight='bold', pad=20)
        ax2.axis('off')

        # -------------------------- 第3列：Model Output（模型输出） --------------------------
        ax3 = fig.add_subplot(gs[row_idx, 2])
        im3 = ax3.imshow(output_img, cmap='gray', aspect='equal')
        if row_idx == 0:
            ax3.set_title(col_titles[2], fontsize=14, fontweight='bold', pad=20)
        ax3.axis('off')

        # -------------------------- 第4列：差异热力图 --------------------------
        ax4 = fig.add_subplot(gs[row_idx, 3])
        im4 = ax4.imshow(diff_heatmap, cmap=cmap_heat, aspect='equal', vmin=0, vmax=1)
        if row_idx == 0:
            ax4.set_title(col_titles[3], fontsize=14, fontweight='bold', pad=20)
        ax4.axis('off')

        # 为热力图添加颜色条（仅最后一行，避免重复）
        if row_idx == len(selected_samples) - 1:
            cbar = plt.colorbar(im4, ax=ax4, shrink=0.8, pad=0.05)
            cbar.set_label('Normalized Difference (0-1)', fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        # 每行左侧添加样本信息（文件名+指标）
        info_text = f"{label}\nFile: {filename}\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"
        fig.text(0.01, 0.97 - row_idx/4, info_text, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))

    # 总标题
    fig.suptitle('Selected Test Samples: Top 2 SSIM & Top 2 PSNR',
                 fontsize=18, fontweight='bold', y=0.99)

    # 保存高清组图
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  高清组图已保存：{save_path}")

# -------------------------- 核心测试函数 --------------------------
def test_model(args):
    # 1. 设备配置
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

    # 2. 创建保存目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    csv_save_path = os.path.join(args.result_dir, "test_metrics.csv")
    group_img_save_path = os.path.join(args.img_save_dir, "top2_ssim_top2_psnr_samples.png")

    # 3. 加载测试集（默认PNG，如需HDF5请注释下面3行，取消HDF5加载代码）
    print("\n[1/5] 加载测试集...")
    test_pairs = get_pair_list(args.data_dir, split="test")
    test_dataset = CTDataset(test_pairs, target_size=256)
    
    # -------------------------- 如需HDF5测试集，替换为以下代码 --------------------------
    # hdf5_path = os.path.join(args.data_dir, "test_dataset.h5")
    # test_dataset = CTHDF5Dataset(hdf5_path, target_size=256)
    # test_pairs = [(f"sample_{i}", f"sample_{i}") for i in range(len(test_dataset))]  # 虚拟配对列表，用于文件名
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"  测试集样本数：{len(test_dataset)} 张")
    print(f"  测试集批次：{len(test_loader)} 批")
    print(f"  图像尺寸：{test_dataset[0][0].shape}")

    # 4. 初始化模型与指标计算器
    print("\n[2/5] 初始化模型与工具...")
    model = LDCTNet_Swin_improve().to(device)
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

    # 5. 批量测试与指标记录
    print("\n[3/5] 开始批量测试...")
    test_results = []
    progress_bar = tqdm(test_loader, desc="[Test] Batch Processing", unit="batch")

    for batch_idx, (ld_imgs, nd_imgs) in enumerate(progress_bar):
        ld_imgs = ld_imgs.to(device, non_blocking=True)
        nd_imgs = nd_imgs.to(device, non_blocking=True)
        batch_size = ld_imgs.size(0)

        with torch.no_grad():
            outputs = model(ld_imgs)  # 模型推理

        # 逐张图像计算指标并保存信息
        for idx in range(batch_size):
            # 提取单张图像
            ld_img = ld_imgs[idx:idx+1].clone()
            nd_img = nd_imgs[idx:idx+1].clone()
            output_img = outputs[idx:idx+1].clone()

            # 计算指标
            metrics = metrics_fn(output_img, nd_img)
            psnr = round(metrics["psnr"].item() if hasattr(metrics["psnr"], 'item') else metrics["psnr"], 4)
            ssim = round(metrics["ssim"].item() if hasattr(metrics["ssim"], 'item') else metrics["ssim"], 4)

            # 获取文件名（适配PNG和HDF5）
            sample_idx = batch_idx * args.batch_size + idx
            if len(test_pairs) > 0 and isinstance(test_pairs[0][0], str):
                filename = os.path.basename(test_pairs[sample_idx][0])
            else:
                filename = f"test_sample_{sample_idx:04d}.png"

            # 保存结果
            test_results.append({
                "filename": filename,
                "psnr": psnr,
                "ssim": ssim,
                "ld_img": ld_img,
                "nd_img": nd_img,
                "output_img": output_img
            })

        # 进度条更新
        current_avg = {
            "psnr": np.mean([r["psnr"] for r in test_results]),
            "ssim": np.mean([r["ssim"] for r in test_results])
        }
        progress_bar.set_postfix({
            "avg_psnr": f"{current_avg['psnr']:.2f}",
            "avg_ssim": f"{current_avg['ssim']:.4f}"
        })

    # 6. 保存完整指标到CSV
    print("\n[4/5] 保存测试指标CSV...")
    csv_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["ld_img", "nd_img", "output_img"]} 
                          for r in test_results])
    # 添加平均值行
    avg_row = {
        "filename": "average",
        "psnr": round(csv_df["psnr"].mean(), 4),
        "ssim": round(csv_df["ssim"].mean(), 4)
    }
    csv_df = pd.concat([csv_df, pd.DataFrame([avg_row])], ignore_index=True)
    csv_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")
    print(f"  完整指标CSV已保存：{csv_save_path}")

    # 7. 筛选2张SSIM最好 + 2张PSNR最好的样本（去重）
    print("\n[5/5] 筛选样本并生成组图...")
    # 筛选Top2 SSIM
    top2_ssim = sorted(test_results, key=lambda x: x["ssim"], reverse=True)[:2]
    # 筛选Top2 PSNR（排除已选的SSIM样本）
    remaining_for_psnr = [r for r in test_results if r not in top2_ssim]
    top2_psnr = sorted(remaining_for_psnr, key=lambda x: x["psnr"], reverse=True)[:2]
    # 组合最终样本（共4张）
    selected_samples = top2_ssim + top2_psnr

    # 生成组图
    create_selected_group_plot(selected_samples, group_img_save_path, dpi=args.dpi)

    # 打印筛选结果总结
    print("\n筛选样本总结：")
    print("=" * 60)
    for i, sample in enumerate(top2_ssim, 1):
        print(f"Top {i} SSIM: {sample['filename']} | SSIM: {sample['ssim']:.4f} | PSNR: {sample['psnr']:.2f} dB")
    for i, sample in enumerate(top2_psnr, 1):
        print(f"Top {i} PSNR: {sample['filename']} | PSNR: {sample['psnr']:.2f} dB | SSIM: {sample['ssim']:.4f}")
    print("=" * 60)

# -------------------------- 命令行参数配置 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--data_dir", type=str, default="./ND_LD_Paired_Data_0.5", 
                        help="数据集根目录（PNG：含test子文件夹；HDF5：直接指向文件夹）")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model_swin.pth", 
                        help="训练好的模型路径")
    parser.add_argument("--result_dir", type=str, default="./test_results", 
                        help="测试指标CSV保存目录")
    parser.add_argument("--img_save_dir", type=str, default="./test_image_floder", 
                        help="测试组图保存目录")
    # 测试配置
    parser.add_argument("--batch_size", type=int, default=1, help="测试批量大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号（-1表示CPU）")
    parser.add_argument("--dpi", type=int, default=600, help="组图分辨率")
    args = parser.parse_args()

    # 执行测试
    test_model(args)