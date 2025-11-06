import h5py

# 替换为你的HDF5文件路径（复制错误提示中的路径）
hdf5_path = "/Users/lxxxx/Desktop/CODE/SwinCT/ground_truth_test/ground_truth_test_000.hdf5"

def print_hdf5_full_structure(hdf5_path):
    """完整打印HDF5文件的所有分组、数据集和形状"""
    with h5py.File(hdf5_path, "r") as f:
        print(f"=== HDF5文件结构：{hdf5_path} ===")
        def traverse(obj, path=""):
            # 打印当前对象的路径和类型
            if isinstance(obj, h5py.Group):
                print(f"[分组] {path}")
                # 递归遍历子分组/数据集
                for key in sorted(obj.keys()):
                    traverse(obj[key], path + "/" + key)
            elif isinstance(obj, h5py.Dataset):
                # 数据集（重点关注形状为 (128, H, W) 或 (H, W, 128) 的）
                print(f"[数据集] {path} | 形状：{obj.shape} | 类型：{obj.dtype}")
        traverse(f)

# 运行查看结构
print_hdf5_full_structure(hdf5_path)