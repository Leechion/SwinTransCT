import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_training_metrics_separate(csv_path, save_dir="sci_metrics_plots_separate", figsize=(16, 12), dpi=600):
    """
    SCI风格训练/验证指标绘图（train和val分离，优化线条粗细，提升观感）
    Args:
        csv_path: CSV指标文件路径（training_metrics.csv）
        save_dir: 图表保存目录
        figsize: 图表尺寸（适配SCI论文多子图布局）
        dpi: 分辨率（SCI要求≥600）
    """
    # 1. 读取并校验CSV文件
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到CSV文件：{csv_path}")
    
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['epoch'], keep='last').reset_index(drop=True)
    print(f"成功读取CSV文件，共 {len(df)} 个有效epoch的指标")

    # 2. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 3. SCI核心样式配置（优化字体大小比例，更协调）
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 期刊标准字体
        'font.size': 9,                    # 基础字体缩小，避免遮挡
        'axes.unicode_minus': False,
        'axes.linewidth': 0.9,             # 坐标轴线条稍纤细，更柔和
        'grid.linewidth': 0.4,             # 网格线弱化，不抢焦点
        'legend.frameon': True,
        'legend.framealpha': 0.95,         # 图例背景更透明
        'legend.edgecolor': 'gray',        # 图例边框灰色，不突兀
        'legend.fontsize': 8,              # 图例字体缩小
        'legend.labelspacing': 0.3,        # 图例条目间距缩小
    })

    # 4. 优化线条与标记样式（核心调整：纤细清晰，提升观感）
    styles = {
        'train': {
            'color': '#0072B2',            # 训练集：深蓝色（稳定专业）
            'linewidth': 1.2,              # 线条粗细优化（原1.6→1.2，更纤细）
            'marker': 'o',                 # 圆形标记
            'markersize': 3.0,             # 标记点缩小（原4.0→3.0，不拥挤）
            'markevery': max(1, len(df)//20),  # 间隔显示标记点（避免epoch多时空叠）
            'alpha': 0.9                   # 透明度适中，避免过浓
        },
        'val': {
            'color': '#D55E00',            # 验证集：深橙色（醒目不刺眼）
            'linewidth': 1.2,              # 统一线条粗细
            'marker': 's',                 # 方形标记
            'markersize': 3.0,             # 标记点缩小
            'markevery': max(1, len(df)//20),  # 间隔显示标记点
            'alpha': 0.9
        }
    }

    # 5. 创建4×2子图布局（保持紧凑，优化子图间距）
    fig, axes = plt.subplots(4, 2, figsize=figsize, constrained_layout=True,
                             gridspec_kw={'wspace': 0.2, 'hspace': 0.3})  # 调整子图间距
    fig.suptitle('Training and Validation Metrics During Model Training', 
                 fontsize=13, fontweight='bold', y=1.01)  # 总标题字体稍小

    # 6. 指标配置（保持不变）
    metrics_config = [
        (0, 'Loss', 'train_loss', 'val_loss', 'Loss Value'),
        (1, 'PSNR', 'train_psnr', 'val_psnr', 'PSNR (dB)'),
        (2, 'SSIM', 'train_ssim', 'val_ssim', 'SSIM'),
        (3, 'RMSE', 'train_rmse', 'val_rmse', 'RMSE')
    ]

    # 7. 逐个绘制子图（优化细节，提升观感）
    for row, metric_name, train_col, val_col, ylabel in metrics_config:
        # ---------------------- 绘制训练集子图（左列）----------------------
        ax_train = axes[row, 0]
        ax_train.plot(
            df['epoch'], df[train_col],
            **styles['train'],  # 应用训练集样式
            label='Training'
        )
        # 子图配置优化
        ax_train.set_title(f'{metric_name} (Training)', fontsize=10, fontweight='bold', pad=8)
        ax_train.set_xlabel('Epoch', fontsize=8.5)
        ax_train.set_ylabel(ylabel, fontsize=8.5)
        # 坐标轴优化（更精细）
        x_margin = max(1, len(df['epoch']) * 0.03)  # 缩小x轴留白（原5%→3%）
        ax_train.set_xlim(left=df['epoch'].min() - x_margin, right=df['epoch'].max() + x_margin)
        y_min = df[train_col].min()
        y_max = df[train_col].max()
        y_margin = (y_max - y_min) * 0.03  # 缩小y轴留白
        ax_train.set_ylim(bottom=y_min - y_margin, top=y_max + y_margin)
        # 网格与边框（更柔和）
        ax_train.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.3)
        ax_train.spines['top'].set_visible(False)
        ax_train.spines['right'].set_visible(False)
        ax_train.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        # 刻度优化（更纤细）
        ax_train.tick_params(axis='both', which='major', direction='in', 
                             length=3, width=0.7, labelsize=8)

        # ---------------------- 绘制验证集子图（右列）----------------------
        ax_val = axes[row, 1]
        ax_val.plot(
            df['epoch'], df[val_col],
            **styles['val'],  # 应用验证集样式
            label='Validation'
        )
        # 子图配置优化
        ax_val.set_title(f'{metric_name} (Validation)', fontsize=10, fontweight='bold', pad=8)
        ax_val.set_xlabel('Epoch', fontsize=8.5)
        ax_val.set_ylabel(ylabel, fontsize=8.5)
        # 坐标轴与训练集对齐（便于对比）
        ax_val.set_xlim(ax_train.get_xlim())
        y_min_val = df[val_col].min()
        y_max_val = df[val_col].max()
        y_margin_val = (y_max_val - y_min_val) * 0.03
        ax_val.set_ylim(bottom=y_min_val - y_margin_val, top=y_max_val + y_margin_val)
        # 网格与边框
        ax_val.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.3)
        ax_val.spines['top'].set_visible(False)
        ax_val.spines['right'].set_visible(False)
        ax_val.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        # 刻度优化
        ax_val.tick_params(axis='both', which='major', direction='in', 
                           length=3, width=0.7, labelsize=8)

    # 8. 保存SCI标准格式文件（保持高分辨率）
    # PNG格式（预览用，600 DPI）
    png_path = os.path.join(save_dir, 'training_val_metrics_separate_sci.png')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"SCI风格PNG图已保存：{png_path}")

    # EPS矢量图（论文提交用）
    eps_path = os.path.join(save_dir, 'training_val_metrics_separate_sci.eps')
    plt.savefig(eps_path, format='eps', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"SCI标准EPS矢量图已保存：{eps_path}")

    # 可选：显示图表（本地调试）
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCI-style plots (train/val separated, optimized style)')
    parser.add_argument('--csv_path', type=str, default='./logs/training_metrics.csv',
                        help='Path to training_metrics.csv (e.g., ./training_metrics.csv)')
    parser.add_argument('--save_dir', type=str, default='./sci_metrics_separate',
                        help='Directory to save optimized plots (default: sci_metrics_separate_optimized)')
    parser.add_argument('--figsize', type=tuple, default=(16, 12),
                        help='Figure size (width, height) for 4×2 subplots (default: (16,12))')
    parser.add_argument('--dpi', type=int, default=600,
                        help='DPI for raster images (SCI requires ≥600, default: 600)')

    args = parser.parse_args()
    
    plot_training_metrics_separate(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        figsize=args.figsize,
        dpi=args.dpi
    )