import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm  # 进度条显示

# --------------------------
# 1. 配置参数（根据需求调整）
# --------------------------
HDF5_FOLDER = "/Users/lxxxx/Desktop/CODE/SwinCT/ground_truth_test"  # 替换为你的28个HDF5文件所在文件夹
OUTPUT_FOLDER = "/Users/lxxxx/Desktop/CODE/SwinCT/ND_LD_Paired_Data"  # 输出配对数据的根文件夹
OCCLUSION_RATE = 0.5  # 遮挡率（0.5=50%剂量降低，可调整为0.3/0.7等）
NOISE_STD = 0.03  # 低剂量噪声强度（剂量越低，噪声越大，建议0.02-0.05）
HOLE_SIZE = (5, 5)  # 挡板孔洞大小（像素），模拟物理挡板尺寸
CT_WINDOW = (40, 400)  # CT窗宽窗位（软组织窗：中心40，宽度400，可按需调整）
IMAGE_SIZE = (256, 256)  # 输出图像尺寸（适配神经网络，可改为512/128等）

# 创建输出文件夹（ND和LD分开存储）
nd_output_dir = os.path.join(OUTPUT_FOLDER, "ND")
ld_output_dir = os.path.join(OUTPUT_FOLDER, "LD")
os.makedirs(nd_output_dir, exist_ok=True)
os.makedirs(ld_output_dir, exist_ok=True)

# --------------------------
# 2. 工具函数（CT窗处理、挡板模拟、批量读取）
# --------------------------
def set_ct_window(img_hu, window_center, window_width):
    """归一化数据无需窗处理，直接返回（避免裁剪为空白）"""
    # 若数据是 0~1：直接归一化到0~1（保持不变）
    if img_hu.max() <= 1.0 and img_hu.min() >= 0.0:
        return np.clip(img_hu, 0.0, 1.0)
    # 若数据是 -1~1：映射到0~1
    elif img_hu.max() <= 1.0 and img_hu.min() >= -1.0:
        return (img_hu + 1.0) / 2.0  # -1→0，1→1
    # 若还是HU值，按原窗处理
    else:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img_windowed = np.clip(img_hu, min_val, max_val)
        img_windowed = (img_windowed - min_val) / (max_val - min_val)
        return img_windowed

def generate_random_mask(height, width, occlusion_rate, hole_size):
    """生成随机挡板模板（适配单张图像尺寸）"""
    # 基础随机二进制模板（0=遮挡，1=通透）
    np.random.seed(None)  # 不固定种子，每张图挡板分布不同（增强泛化性）
    mask = np.random.choice([0, 1], size=(height, width), p=[occlusion_rate, 1-occlusion_rate])
    mask = mask.astype(np.uint8)
    # 形态学处理：平滑孔洞，模拟真实挡板
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, hole_size)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)
    return mask.astype(np.float32)

def simulate_low_dose_ct(nd_img_hu, mask, noise_std):
    """将标准剂量CT（HU值）模拟为低剂量CT（挡板+噪声）"""
    # 步骤1：CT窗处理（转为0-1的灰度图）
    nd_img_norm = set_ct_window(nd_img_hu, *CT_WINDOW)
    # 步骤2：挡板遮挡（模拟稀疏信号）
    sparse_img = nd_img_norm * mask
    # 步骤3：添加低剂量统计噪声
    noise = np.random.normal(0, noise_std, size=nd_img_norm.shape)
    ld_img_norm = sparse_img + noise
    # 步骤4：裁剪溢出值，确保0-1范围
    ld_img_norm = np.clip(ld_img_norm, 0.0, 1.0)
    # 步骤5：轻微平滑（模拟低剂量CT的噪声特性，可选）
    ld_img_norm = cv2.GaussianBlur(ld_img_norm, (3, 3), sigmaX=0.8)
    return nd_img_norm, ld_img_norm

def read_hdf5_ct_images(hdf5_path):
    """读取单个HDF5文件中的所有128张NDCT图像（HU值）"""
    ct_images_hu = []
    with h5py.File(hdf5_path, "r") as f:
        # --------------------------
        # 关键：根据你的HDF5结构修改路径！
        # 假设CT图像（HU值）存储在 "/reconstructed_images" 或 "/ct_images" 路径下
        # 若不确定，先运行之前的HDF5结构遍历代码确认！
        # --------------------------
        # 示例1：如果图像存储在分组下的数据集（常见科研结构）
        if "/data" in f:
            img_dataset = f["/data"]
        elif "/ct_images" in f:
            img_dataset = f["/ct_images"]
        elif "/images" in f:
            img_dataset = f["/images"]
        else:
            raise ValueError(f"HDF5文件 {hdf5_path} 中未找到CT图像数据集！请确认路径。")
        
        # 读取所有图像（假设数据集形状为 (128, H, W) → (数量, 高度, 宽度)）
        if img_dataset.shape[0] == 128:
            ct_images_hu = img_dataset[:]  # (128, H, W)
        else:
            # 若形状为 (H, W, 128)，转置为 (128, H, W)
            ct_images_hu = np.transpose(img_dataset, (2, 0, 1))
        
        # 新增：打印数据的数值范围（关键！）
        print(f"\n{os.path.basename(hdf5_path)} 数据范围：")
        print(f"最小值：{ct_images_hu.min():.2f} | 最大值：{ct_images_hu.max():.2f}")
        print(f"平均值：{ct_images_hu.mean():.2f}")

        # 确保数据是HU值（若HDF5存储的是归一化后的数据，需注释以下校准步骤）
        # 假设HDF5中存储了校准参数（斜率/截距），若没有则手动设置（如 slope=1, intercept=-1024）
        if "rescale_slope" in img_dataset.attrs and "rescale_intercept" in img_dataset.attrs:
            slope = img_dataset.attrs["rescale_slope"]
            intercept = img_dataset.attrs["rescale_intercept"]
            ct_images_hu = ct_images_hu * slope + intercept  # 转换为HU值
    
    return ct_images_hu

# --------------------------
# 3. 批量处理主流程
# --------------------------
def batch_process_hdf5_to_paired_nd_ld():
    # 获取所有HDF5文件（按文件名排序，确保顺序一致）
    hdf5_files = [f for f in os.listdir(HDF5_FOLDER) if f.endswith(".h5") or f.endswith(".hdf5")]
    hdf5_files.sort()  # 排序，避免处理顺序混乱
    print(f"找到 {len(hdf5_files)} 个HDF5文件，开始批量处理...")
    
    total_img_count = 0  # 统计总图像数（28×128=3584张）
    
    # 遍历每个HDF5文件
    for hdf5_idx, hdf5_file in enumerate(hdf5_files):
        hdf5_path = os.path.join(HDF5_FOLDER, hdf5_file)
        print(f"\n处理第 {hdf5_idx+1}/{len(hdf5_files)} 个文件：{hdf5_file}")
        
        # 读取当前HDF5中的128张CT图像（HU值）
        try:
            ct_images_hu = read_hdf5_ct_images(hdf5_path)
            assert len(ct_images_hu) == 128, f"该HDF5文件仅包含 {len(ct_images_hu)} 张图像，不符合128张要求！"
        except Exception as e:
            print(f"警告：读取 {hdf5_file} 失败！错误：{str(e)}，跳过该文件。")
            continue
        
        # 遍历当前HDF5中的每张图像
        for img_idx, img_hu in tqdm(enumerate(ct_images_hu), total=128, desc="处理图像"):
            # 生成当前图像的专属随机挡板模板（每张图挡板不同）
            mask = generate_random_mask(img_hu.shape[0], img_hu.shape[1], OCCLUSION_RATE, HOLE_SIZE)
            
            # 模拟低剂量CT
            nd_img_norm, ld_img_norm = simulate_low_dose_ct(img_hu, mask, NOISE_STD)
            
            # 调整图像尺寸（适配神经网络）
            nd_img_resized = cv2.resize(nd_img_norm, IMAGE_SIZE)
            ld_img_resized = cv2.resize(ld_img_norm, IMAGE_SIZE)
            
            # 生成配对文件名（格式：nd_0001.png, ld_0001.png，按总计数排序）
            total_img_count += 1
            img_filename = f"{total_img_count:04d}.png"  # 4位数字编号（0001~3584）
            nd_save_path = os.path.join(nd_output_dir, img_filename)
            ld_save_path = os.path.join(ld_output_dir, img_filename)
            
            # 转换为8位灰度图（0-255）并保存
            nd_img_uint8 = (nd_img_resized * 255).astype(np.uint8)
            ld_img_uint8 = (ld_img_resized * 255).astype(np.uint8)
            cv2.imwrite(nd_save_path, nd_img_uint8)
            cv2.imwrite(ld_save_path, ld_img_uint8)
    
    print(f"\n批量处理完成！")
    print(f"总生成配对图像数：{total_img_count} 对（ND和LD各 {total_img_count} 张）")
    print(f"ND图像路径：{nd_output_dir}")
    print(f"LD图像路径：{ld_output_dir}")

# --------------------------
# 4. 执行批量处理
# --------------------------
if __name__ == "__main__":
    batch_process_hdf5_to_paired_nd_ld()