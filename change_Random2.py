import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm  # 进度条显示

# --------------------------
# 1. 配置参数（重点调整黑斑相关参数）
# --------------------------
HDF5_FOLDER = "/Users/lxxxx/Desktop/ground_truth_test"  # 替换为你的HDF5文件所在文件夹
OUTPUT_FOLDER = "./ND_LD_Paired_Data_Large_Black_Spots"  # 输出配对数据的根文件夹
OCCLUSION_RATE = 0.4  # 总遮挡率（保持低剂量特性，0.6~0.75为宜）
NOISE_STD = 0.03  # 低剂量噪声强度（建议0.02~0.05）
MAX_HOLE_COUNT = 40  # 最大孔洞数量（核心！值越小，黑斑越少越大；8~15可调）
CT_WINDOW = (40, 400)  # CT窗宽窗位（软组织窗：中心40，宽度400）
IMAGE_SIZE = (256, 256)  # 输出图像尺寸（适配神经网络）

# 创建输出文件夹（ND和LD分开存储）
nd_output_dir = os.path.join(OUTPUT_FOLDER, "ND")
ld_output_dir = os.path.join(OUTPUT_FOLDER, "LD")
os.makedirs(nd_output_dir, exist_ok=True)
os.makedirs(ld_output_dir, exist_ok=True)

# --------------------------
# 2. 工具函数（核心修改：generate_random_mask 生成不规则大块黑斑）
# --------------------------
def set_ct_window(img_hu, window_center, window_width):
    """归一化数据无需窗处理，直接返回（避免裁剪为空白）"""
    if img_hu.max() <= 1.0 and img_hu.min() >= 0.0:
        return np.clip(img_hu, 0.0, 1.0)
    elif img_hu.max() <= 1.0 and img_hu.min() >= -1.0:
        return (img_hu + 1.0) / 2.0  # -1→0，1→1
    else:
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img_windowed = np.clip(img_hu, min_val, max_val)
        img_windowed = (img_windowed - min_val) / (max_val - min_val)
        return img_windowed

def generate_random_mask(height, width, occlusion_rate, max_hole_count):
    """
    生成「少而大的不规则黑斑」掩码（核心函数）
    逻辑：固定少量孔洞，通过随机半径+不规则边缘生成大块通透区，遮挡区自然形成大块黑斑
    """
    np.random.seed(None)  # 每张图黑斑分布不同，增强泛化性
    mask = np.zeros((height, width), dtype=np.uint8)  # 初始全遮挡（0=遮挡，1=通透）
    total_pixels = height * width
    target_open_pixels = total_pixels * (1 - occlusion_rate)  # 目标通透像素数（由遮挡率决定）
    open_pixels = 0  # 当前已通透的像素数

    # 生成随机数量的大孔洞（数量在 max_hole_count//2 ~ max_hole_count 之间，确保少而大）
    actual_hole_count = np.random.randint(max_hole_count // 2, max_hole_count)
    
    for _ in range(actual_hole_count):
        if open_pixels >= target_open_pixels:
            break  # 已达到目标通透率，停止生成孔洞
        
        # 步骤1：随机选择孔洞中心（避免超出图像边界）
        center_x = np.random.randint(width // 10, width - width // 10)
        center_y = np.random.randint(height // 10, height - height // 10)
        
        # 步骤2：随机生成大半径（基于图像尺寸自适应，确保孔洞足够大）
        min_radius = min(height, width) // 15  # 最小半径（图像尺寸的1/15）
        max_radius = min(height, width) // 5   # 最大半径（图像尺寸的1/5）
        radius = np.random.randint(min_radius, max_radius)
        
        # 步骤3：生成不规则圆形孔洞（避免完美圆形，更贴近真实遮挡）
        y, x = np.ogrid[:height, :width]
        # 计算到中心的距离
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        # 随机偏移半径，让孔洞边缘不规则（0.7~1.3倍随机缩放）
        random_radius_offset = np.random.uniform(0.7, 1.3, size=dist_from_center.shape)
        dist_from_center = dist_from_center * random_radius_offset
        
        # 步骤4：标记当前孔洞的通透区域
        hole_mask = (dist_from_center <= radius).astype(np.uint8)
        # 计算新增的通透像素数（避免与已有孔洞重叠）
        new_open_pixels = np.sum(hole_mask) - np.sum(mask & hole_mask)
        
        # 步骤5：避免孔洞过大导致总通透率超标
        if open_pixels + new_open_pixels > target_open_pixels:
            # 按比例裁剪孔洞，确保不超过目标通透率
            excess_ratio = (target_open_pixels - open_pixels) / new_open_pixels
            if excess_ratio > 0:
                # 随机保留部分孔洞像素
                hole_coords = np.where(hole_mask == 1)
                keep_count = int(len(hole_coords[0]) * excess_ratio)
                keep_indices = np.random.choice(len(hole_coords[0]), size=keep_count, replace=False)
                temp_mask = np.zeros_like(hole_mask)
                temp_mask[hole_coords[0][keep_indices], hole_coords[1][keep_indices]] = 1
                mask |= temp_mask
                open_pixels = target_open_pixels  # 刚好达到目标，跳出循环
            break
        
        # 步骤6：合并当前孔洞到总掩码
        mask |= hole_mask
        open_pixels += new_open_pixels

    # 步骤7：形态学处理（让孔洞边缘更柔和，黑斑过渡自然）
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, smooth_kernel, iterations=1)  # 膨胀：孔洞边缘更圆润
    mask = cv2.erode(mask, smooth_kernel, iterations=1)   # 腐蚀：还原尺寸，去除锯齿

    return mask.astype(np.float32)

def simulate_low_dose_ct(nd_img_hu, mask, noise_std):
    """将标准剂量CT（HU值）模拟为低剂量CT（大块黑斑+噪声）"""
    # 步骤1：CT窗处理（转为0-1的灰度图）
    nd_img_norm = set_ct_window(nd_img_hu, *CT_WINDOW)
    # 步骤2：挡板遮挡（生成大块黑斑）
    sparse_img = nd_img_norm * mask
    # 步骤3：添加低剂量统计噪声（泊松噪声更贴近真实，可选替换高斯噪声）
    # 泊松噪声版本（注释高斯噪声，启用以下2行）
    # poisson_noise = np.random.poisson(lam=sparse_img * 200 * noise_std) / 200
    # ld_img_norm = sparse_img + poisson_noise
    # 高斯噪声版本（保持原逻辑）
    noise = np.random.normal(0, noise_std, size=nd_img_norm.shape)
    ld_img_norm = sparse_img + noise
    # 步骤4：裁剪溢出值，确保0-1范围
    ld_img_norm = np.clip(ld_img_norm, 0.0, 1.0)
    # 步骤5：轻微平滑（避免噪声过于杂乱，不影响大块黑斑）
    ld_img_norm = cv2.GaussianBlur(ld_img_norm, (3, 3), sigmaX=0.8)
    return nd_img_norm, ld_img_norm

def read_hdf5_ct_images(hdf5_path):
    """读取单个HDF5文件中的所有128张NDCT图像（HU值）"""
    ct_images_hu = []
    with h5py.File(hdf5_path, "r") as f:
        # 自动识别常见的CT图像数据集路径
        if "/data" in f:
            img_dataset = f["/data"]
        elif "/ct_images" in f:
            img_dataset = f["/ct_images"]
        elif "/images" in f:
            img_dataset = f["/images"]
        elif "/reconstructed_images" in f:
            img_dataset = f["/reconstructed_images"]
        else:
            raise ValueError(f"HDF5文件 {hdf5_path} 中未找到CT图像数据集！请确认路径。")
        
        # 读取所有图像（适配 (128, H, W) 或 (H, W, 128) 格式）
        if img_dataset.shape[0] == 128:
            ct_images_hu = img_dataset[:]  # (128, H, W)
        else:
            ct_images_hu = np.transpose(img_dataset, (2, 0, 1))  # (H, W, 128) → (128, H, W)
        
        # 打印数据范围，帮助用户确认数据合理性
        print(f"\n{os.path.basename(hdf5_path)} 数据范围：")
        print(f"最小值：{ct_images_hu.min():.2f} | 最大值：{ct_images_hu.max():.2f} | 平均值：{ct_images_hu.mean():.2f}")

        # HU值校准（若HDF5包含校准参数）
        if "rescale_slope" in img_dataset.attrs and "rescale_intercept" in img_dataset.attrs:
            slope = img_dataset.attrs["rescale_slope"]
            intercept = img_dataset.attrs["rescale_intercept"]
            ct_images_hu = ct_images_hu * slope + intercept  # 转换为HU值
    
    return ct_images_hu

# --------------------------
# 3. 批量处理主流程（无需修改，直接调用新的掩码生成函数）
# --------------------------
def batch_process_hdf5_to_paired_nd_ld():
    # 获取所有HDF5文件（按文件名排序，确保配对顺序一致）
    hdf5_files = [f for f in os.listdir(HDF5_FOLDER) if f.endswith(".h5") or f.endswith(".hdf5")]
    hdf5_files.sort()
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
            # 生成「少而大的不规则黑斑」掩码（调用修改后的函数）
            mask = generate_random_mask(
                height=img_hu.shape[0],
                width=img_hu.shape[1],
                occlusion_rate=OCCLUSION_RATE,
                max_hole_count=MAX_HOLE_COUNT
            )
            
            # 模拟低剂量CT（大块黑斑+噪声）
            nd_img_norm, ld_img_norm = simulate_low_dose_ct(img_hu, mask, NOISE_STD)
            
            # 调整图像尺寸（适配神经网络）
            nd_img_resized = cv2.resize(nd_img_norm, IMAGE_SIZE)
            ld_img_resized = cv2.resize(ld_img_norm, IMAGE_SIZE)
            
            # 生成配对文件名（按总计数排序，确保ND和LD一一对应）
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
    print(f"黑斑特性：少而大（孔洞数量 {MAX_HOLE_COUNT//2}~{MAX_HOLE_COUNT} 个，总遮挡率 {OCCLUSION_RATE}）")

# --------------------------
# 4. 执行批量处理
# --------------------------
if __name__ == "__main__":
    batch_process_hdf5_to_paired_nd_ld()