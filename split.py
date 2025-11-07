import os
import shutil
import numpy as np
from tqdm import tqdm

# --------------------------
# 1. 配置参数（请确认路径正确！）
# --------------------------
# 原始配对数据路径（批量处理生成的ND和LD文件夹）
ND_RAW_DIR = "./ND_LD_Paired_Data/ND"
LD_RAW_DIR = "./ND_LD_Paired_Data/LD"

# 划分后的数据保存根路径（与原始数据同目录，会自动创建train/val/test）
OUTPUT_ROOT = "/Users/lxxxx/Desktop/CODE/SwinCT/ND_LD_Paired_Data"

# 划分比例（8:1:1）
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子（固定，确保可复现）
SEED = 42
np.random.seed(SEED)

# --------------------------
# 2. 修复：强制创建所有目标文件夹（关键！）
# --------------------------
def create_all_dirs():
    """提前创建所有需要的文件夹，避免FileNotFoundError"""
    # 定义所有需要的文件夹路径
    required_dirs = [
        os.path.join(OUTPUT_ROOT, "train", "ND"),
        os.path.join(OUTPUT_ROOT, "train", "LD"),
        os.path.join(OUTPUT_ROOT, "val", "ND"),
        os.path.join(OUTPUT_ROOT, "val", "LD"),
        os.path.join(OUTPUT_ROOT, "test", "ND"),
        os.path.join(OUTPUT_ROOT, "test", "LD")
    ]
    
    # 循环创建文件夹（exist_ok=True 表示已存在也不报错）
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"已确保文件夹存在：{dir_path}")
    print("\n所有目标文件夹创建完成！")

# --------------------------
# 3. 复制配对文件（逻辑不变）
# --------------------------
def copy_paired_images(image_filenames, split_type):
    """复制ND和LD配对图像到对应文件夹"""
    # 拼接源路径和目标路径
    nd_src_dir = ND_RAW_DIR
    ld_src_dir = LD_RAW_DIR
    nd_dst_dir = os.path.join(OUTPUT_ROOT, split_type, "ND")
    ld_dst_dir = os.path.join(OUTPUT_ROOT, split_type, "LD")
    
    print(f"\n开始复制{split_type}集（{len(image_filenames)}对图像）...")
    for filename in tqdm(image_filenames):
        # 复制ND图像
        nd_src = os.path.join(nd_src_dir, filename)
        nd_dst = os.path.join(nd_dst_dir, filename)
        if os.path.exists(nd_src):  # 避免源文件不存在报错
            shutil.copy2(nd_src, nd_dst)
        else:
            print(f"警告：ND源文件不存在 → {nd_src}")
        
        # 复制对应LD图像（文件名一致）
        ld_src = os.path.join(ld_src_dir, filename)
        ld_dst = os.path.join(ld_dst_dir, filename)
        if os.path.exists(ld_src):
            shutil.copy2(ld_src, ld_dst)
        else:
            print(f"警告：LD源文件不存在 → {ld_src}")

# --------------------------
# 4. 核心划分逻辑（调整执行顺序：先创建文件夹，再划分）
# --------------------------
def split_train_val_test():
    # 第一步：验证原始ND和LD文件夹是否存在，且文件数量一致
    if not os.path.exists(ND_RAW_DIR):
        raise FileNotFoundError(f"原始ND文件夹不存在！路径：{ND_RAW_DIR}")
    if not os.path.exists(LD_RAW_DIR):
        raise FileNotFoundError(f"原始LD文件夹不存在！路径：{LD_RAW_DIR}")
    
    # 统计原始文件数量
    nd_filenames = [f for f in os.listdir(ND_RAW_DIR) if f.endswith(".png")]
    ld_filenames = [f for f in os.listdir(LD_RAW_DIR) if f.endswith(".png")]
    nd_count = len(nd_filenames)
    ld_count = len(ld_filenames)
    
    if nd_count != ld_count:
        raise ValueError(f"ND和LD文件数量不匹配！ND：{nd_count}张，LD：{ld_count}张")
    if nd_count == 0:
        raise ValueError("原始ND文件夹中没有PNG图像！请检查批量处理是否成功。")
    
    print(f"✅ 原始数据验证通过：共{nd_count}对ND-LD图像")
    
    # 第二步：提前创建所有目标文件夹（修复核心）
    create_all_dirs()
    
    # 第三步：随机打乱文件名，保证划分均匀
    shuffled_indices = np.random.permutation(nd_count)
    shuffled_filenames = [nd_filenames[i] for i in shuffled_indices]
    
    # 第四步：计算各集数量（避免小数误差）
    train_count = int(nd_count * TRAIN_RATIO)
    val_count = int(nd_count * VAL_RATIO)
    test_count = nd_count - train_count - val_count  # 剩余归测试集
    
    print(f"\n📊 划分方案：")
    print(f"训练集：{train_count}对 | 验证集：{val_count}对 | 测试集：{test_count}对")
    
    # 第五步：分割文件名列表
    train_filenames = shuffled_filenames[:train_count]
    val_filenames = shuffled_filenames[train_count:train_count+val_count]
    test_filenames = shuffled_filenames[train_count+val_count:]
    
    # 第六步：复制文件
    copy_paired_images(train_filenames, "train")
    copy_paired_images(val_filenames, "val")
    copy_paired_images(test_filenames, "test")
    
    print("\n🎉 划分完成！")
    print(f"训练集：{os.path.join(OUTPUT_ROOT, 'train')}")
    print(f"验证集：{os.path.join(OUTPUT_ROOT, 'val')}")
    print(f"测试集：{os.path.join(OUTPUT_ROOT, 'test')}")

# --------------------------
# 5. 执行划分
# --------------------------
if __name__ == "__main__":
    try:
        split_train_val_test()
    except Exception as e:
        print(f"\n❌ 划分失败！错误：{str(e)}")