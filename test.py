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
from AINDNet import AINDNet  # 新增：导入AINDNet（如果需要测试该模型）
from A_model import LDCTNet_NoFreq  # 新增：导入AINDNet（如果需要测试该模型）
from C_model import LDCTNet_NoResidualFusion  # 新增：导入AINDNet（如果需要测试该模型）
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
    生成最终组图：4行（用户指定的4张图）×4列（ND、LD、Output、差异热力图）
    支持用户指定的自定义标签（如"Sample 1"、"Case A"等）
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

    # 创建4行4列的组图（4张指定图 × 4列）
    fig = plt.figure(figsize=(20, 20), dpi=dpi)
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.2)

    # 子图列标题
    col_titles = ['Normal-Dose CT (ND)', 'Low-Dose CT (LD)', 'Model Output', 'Difference Heatmap (|Output - ND|)']
    # 样本行标签（默认用"Selected Sample X"，用户可通过参数自定义）
    sample_labels = [s.get('label', f'Selected Sample {i + 1}') for i, s in enumerate(selected_samples)]

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
        fig.text(0.01, 0.97 - row_idx / 4, info_text, fontsize=11, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9))

    # 总标题（区分"用户指定"和"自动筛选"）
    if 'is_selected' in selected_samples[0] and selected_samples[0]['is_selected']:
        fig.suptitle('User-Specified Test Samples (4 Selected Cases)',
                     fontsize=18, fontweight='bold', y=0.99)
    else:
        fig.suptitle('Selected Test Samples: Top 2 SSIM & Top 2 PSNR',
                     fontsize=18, fontweight='bold', y=0.99)

    # 保存高清组图
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"  高清组图已保存：{save_path}")


# -------------------------- 新增：解析用户指定的样本（核心功能） --------------------------
def parse_selected_samples(args, test_results, test_pairs, test_dataset_len):
    """
    解析用户指定的4张样本（按文件名或索引），返回有效样本列表
    Args:
        args: 命令行参数（含--selected_files和--selected_indices）
        test_results: 所有测试样本的结果（含filename、ld_img、nd_img等）
        test_pairs: 测试集的文件配对列表（用于文件名映射）
        test_dataset_len: 测试集总样本数
    Returns:
        selected_samples: 4张有效样本（含is_selected标记）
    """
    selected_samples = []
    # 构建「文件名→样本结果」的映射（方便快速查找）
    filename_to_result = {r['filename']: r for r in test_results}
    all_valid_filenames = list(filename_to_result.keys())

    # 1. 优先处理按文件名指定（--selected_files）
    if args.selected_files is not None and len(args.selected_files) > 0:
        print(f"\n[指定样本] 正在加载用户指定的文件名：{args.selected_files}")
        for file in args.selected_files[:4]:  # 最多取前4个
            file_basename = os.path.basename(file)  # 兼容带路径的文件名
            if file_basename in filename_to_result:
                sample = filename_to_result[file_basename]
                sample['is_selected'] = True
                # 支持自定义标签（通过--selected_labels，顺序对应）
                label_idx = args.selected_files.index(file)
                if args.selected_labels is not None and label_idx < len(args.selected_labels):
                    sample['label'] = args.selected_labels[label_idx]
                selected_samples.append(sample)
                print(f"  ✅ 找到样本：{file_basename}")
            else:
                print(f"  ❌ 未找到样本：{file_basename}，可用文件名：{all_valid_filenames[:10]}...")  # 显示前10个可用文件名

    # 2. 处理按索引指定（--selected_indices）（如果文件名指定不足4个）
    if len(selected_samples) < 4 and args.selected_indices is not None and len(args.selected_indices) > 0:
        print(f"\n[指定样本] 正在加载用户指定的索引：{args.selected_indices}")
        for idx in args.selected_indices[:4 - len(selected_samples)]:  # 补充到4个
            if 0 <= idx < test_dataset_len:
                # 按索引找到对应的文件名
                if len(test_pairs) > 0 and isinstance(test_pairs[idx][0], str):
                    filename = os.path.basename(test_pairs[idx][0])
                else:
                    filename = f"test_sample_{idx:04d}.png"
                if filename in filename_to_result:
                    sample = filename_to_result[filename]
                    sample['is_selected'] = True
                    # 支持自定义标签
                    label_idx = args.selected_indices.index(idx)
                    if args.selected_labels is not None and label_idx < len(args.selected_labels):
                        sample['label'] = args.selected_labels[label_idx]
                    selected_samples.append(sample)
                    print(f"  ✅ 找到样本索引 {idx}：{filename}")
                else:
                    print(f"  ❌ 索引 {idx} 对应的样本不存在")
            else:
                print(f"  ❌ 索引 {idx} 超出范围（测试集共 {test_dataset_len} 个样本，索引0-{test_dataset_len - 1}）")

    # 3. 检查是否凑够4个样本（不足则提示并退出）
    if len(selected_samples) < 4:
        raise ValueError(f"\n[错误] 指定的有效样本仅 {len(selected_samples)} 个，需至少4个！")

    return selected_samples


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
    print(f"  指定样本文件名: {args.selected_files if args.selected_files else '无'}")
    print(f"  指定样本索引: {args.selected_indices if args.selected_indices else '无'}")
    print(f"=" * 60)

    # 2. 创建保存目录
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    csv_save_path = os.path.join(args.result_dir, "test_metrics_C.csv")
    group_img_save_path = os.path.join(args.img_save_dir,
                                       "selected_samples.png" if (args.selected_files or args.selected_indices)
                                       else "top2_ssim_top2_psnr_samples.png")

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
    test_dataset_len = len(test_dataset)
    print(f"  测试集样本数：{test_dataset_len} 张")
    print(f"  测试集批次：{len(test_loader)} 批")
    print(f"  图像尺寸：{test_dataset[0][0].shape}")

    # 4. 初始化模型与指标计算器
    print("\n[2/5] 初始化模型与工具...")
    # 支持切换不同模型（根据需要修改）
    if "AINDNet" in args.model_path or args.model == "AINDNet":
        model = AINDNet().to(device)
    elif "LDCTNet256" in args.model_path or args.model == "LDCTNet256":
        model = LDCTNet256().to(device)
    elif "LDCTNet_Swin_improve" in args.model_path or args.model == "LDCTNet_Swin_improve":
        model = LDCTNet_Swin_improve().to(device)
    else:
        model = LDCTNet_NoResidualFusion().to(device)  # 默认模型
    print(f"  模型类型：{model.__class__.__name__}")
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
            # 新增：适配AINDNet的双输出（noise_map, recon_out）
            outputs = model(ld_imgs)
            if isinstance(outputs, tuple) and len(outputs) == 2:  # 判断是否为双输出
                outputs = outputs[1]  # 只取重建图（recon_out）

        # 逐张图像计算指标并保存信息
        for idx in range(batch_size):
            # 提取单张图像
            ld_img = ld_imgs[idx:idx + 1].clone()
            nd_img = nd_imgs[idx:idx + 1].clone()
            output_img = outputs[idx:idx + 1].clone()

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

    # 7. 筛选样本并生成组图（核心修改：支持用户指定）
    print("\n[5/5] 筛选样本并生成组图...")
    if args.selected_files or args.selected_indices:
        # 新增：使用用户指定的4张样本
        selected_samples = parse_selected_samples(args, test_results, test_pairs, test_dataset_len)
    else:
        # 原有逻辑：筛选Top2 SSIM + 2张PSNR最好的样本（去重）
        top2_ssim = sorted(test_results, key=lambda x: x["ssim"], reverse=True)[:2]
        remaining_for_psnr = [r for r in test_results if r not in top2_ssim]
        top2_psnr = sorted(remaining_for_psnr, key=lambda x: x["psnr"], reverse=True)[:2]
        selected_samples = top2_ssim + top2_psnr
        # 标记为自动筛选
        for s in selected_samples:
            s['is_selected'] = False

    # 生成组图
    create_selected_group_plot(selected_samples, group_img_save_path, dpi=args.dpi)

    # 打印筛选结果总结
    print("\n筛选样本总结：")
    print("=" * 60)
    for i, sample in enumerate(selected_samples, 1):
        label = sample.get('label', f'Sample {i}')
        print(f"{label}: {sample['filename']} | PSNR: {sample['psnr']:.2f} dB | SSIM: {sample['ssim']:.4f}")
    print("=" * 60)


# -------------------------- 命令行参数配置（新增指定样本参数） --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径配置
    parser.add_argument("--data_dir", type=str, default="./ND_LD_Paired_Data_0.5",
                        help="数据集根目录（PNG：含test子文件夹；HDF5：直接指向文件夹）")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model_C.pth",
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
    parser.add_argument("--model", type=str, default="LDCTNet_NoFreq",
                        choices=["RED_CNN", "AINDNet", "LDCTNet256", "LDCTNet_Swin_improve", "LDCTNet_NoFreq"],
                        help="指定模型类型（避免自动识别错误）")
    # 新增：指定4张样本的参数（二选一即可，需凑够4张）
    parser.add_argument("--selected_files", nargs="+", type=str,
                        default=['1490.png', '1423.png', '1250.png', '1155.png'],
                        help="指定测试的样本文件名（需是test集内的文件名，最多4个）")
    parser.add_argument("--selected_indices", nargs="+", type=int, default=None,
                        help="指定测试的样本索引（从0开始，如[0,1,2,3]，最多4个）")
    parser.add_argument("--selected_labels", nargs="+", type=str, default=None,
                        help="为指定样本自定义标签（顺序与selected_files/selected_indices对应，如['Case 1', 'Case 2']）")

    args = parser.parse_args()

    # 执行测试
    test_model(args)

    # 0.5 ['3266.png','1601.png','0025.png', '1886.png']
    # 0.7 ['1490.png','1423.png','1250.png', '1155.png']
    # 0.3 ['3210.png','3006.png','1601.png', '0025.png']
