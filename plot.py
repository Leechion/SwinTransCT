import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# 设置中文字体（避免中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows：SimHei / Mac：Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def plot_training_metrics(csv_path, save_dir="metrics_plots", figsize=(15, 10), dpi=300):
    """
    从CSV文件绘制训练/验证指标曲线图
    Args:
        csv_path: CSV指标文件路径（training_metrics.csv）
        save_dir: 图表保存目录
        figsize: 图表尺寸
        dpi: 保存图片的分辨率（300为高清，适合论文）
    """
    # 1. 读取CSV文件
    if not os.path.exists(csv_path):
        print(f"错误：未找到CSV文件 {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"成功读取CSV文件，共 {len(df)} 个epoch的指标")
    print("CSV文件列名：", df.columns.tolist())

    # 2. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 3. 设置图表样式（专业美观）
    plt.style.use('default')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 专业配色方案
    markers = ['o', 's', '^', 'D']  # 标记点样式（可选）
    line_width = 2.0
    marker_size = 4

    # 4. 创建2x2子图（分别显示4个指标）
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('训练过程指标变化曲线', fontsize=16, fontweight='bold', y=0.98)

    # 定义要绘制的指标（与CSV列名对应）
    metrics_config = [
        # (子图位置, 指标名称, 训练集列名, 验证集列名, y轴标签)
        ((0, 0), '损失值 (Loss)', 'train_loss', 'val_loss', '损失值'),
        ((0, 1), '峰值信噪比 (PSNR)', 'train_psnr', 'val_psnr', 'PSNR (dB)'),
        ((1, 0), '结构相似性 (SSIM)', 'train_ssim', 'val_ssim', 'SSIM'),
        ((1, 1), '均方根误差 (RMSE)', 'train_rmse', 'val_rmse', 'RMSE')
    ]

    # 5. 逐个绘制指标曲线
    for (row, col), title, train_col, val_col, ylabel in metrics_config:
        ax = axes[row, col]
        
        # 绘制训练集曲线
        ax.plot(
            df['epoch'], df[train_col],
            color=colors[0], linewidth=line_width, marker=markers[0], markersize=marker_size,
            label=f'训练集', alpha=0.8
        )
        
        # 绘制验证集曲线
        ax.plot(
            df['epoch'], df[val_col],
            color=colors[1], linewidth=line_width, marker=markers[1], markersize=marker_size,
            label=f'验证集', alpha=0.8
        )
        
        # 设置子图标题和标签
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # 添加网格（便于读取数值）
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加图例
        ax.legend(loc='best', fontsize=10)
        
        # 优化坐标轴刻度
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 6. 调整子图间距（避免重叠）
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 7. 保存高清图片（支持PNG/JPG/SVG格式）
    save_path_png = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path_png, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"图表已保存为：{save_path_png}")

    # 可选：保存为SVG矢量图（无限放大不失真，适合论文）
    save_path_svg = os.path.join(save_dir, 'training_metrics.svg')
    plt.savefig(save_path_svg, bbox_inches='tight', facecolor='white')
    print(f"矢量图已保存为：{save_path_svg}")

    # 8. 显示图表（可选）
    plt.show()

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='绘制训练指标曲线图')
    parser.add_argument('--csv_path', type=str, 
                        default='/Users/lxxxx/Desktop/CODE/SwinCT/logs/training_metrics.csv')
    parser.add_argument('--save_dir', type=str, default='metrics_plots', 
                        help='图表保存目录（默认：metrics_plots）')
    parser.add_argument('--figsize', type=tuple, default=(15, 10), 
                        help='图表尺寸（默认：(15,10)）')
    parser.add_argument('--dpi', type=int, default=500, 
                        help='图片分辨率（默认：500，越高越清晰）')

    args = parser.parse_args()
    
    # 运行绘图函数
    plot_training_metrics(
        csv_path=args.csv_path,
        save_dir=args.save_dir,
        figsize=args.figsize,
        dpi=args.dpi
    )