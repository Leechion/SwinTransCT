import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import h5py
import os

class CTDataset(Dataset):
    """CT数据集加载器（适配HDF5格式）：增强 + 标准化 + 高效加载"""

    def __init__(self, hdf5_path, target_size=256, augment=True):
        self.hdf5_path = hdf5_path  # HDF5文件路径（如train_dataset.h5）
        self.target_size = target_size
        self.augment = augment

        # 预加载数据集元信息（无需加载全部数据，节省内存）
        with h5py.File(self.hdf5_path, "r") as f:
            self.num_samples = f.attrs["num_images"]  # 图像对数
            self.image_height = f.attrs["image_height"]  # 原始高度
            self.image_width = f.attrs["image_width"]    # 原始宽度

        # 预定义变换管道（Resize + 转Tensor，HDF5数据已为0-1归一化）
        self.transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BILINEAR),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 关键：部分加载（仅读取当前索引的图像，避免加载整个数据集）
        with h5py.File(self.hdf5_path, "r") as f:
            # 读取ND（高剂量）和LD（低剂量）图像（shape: (H, W)，dtype: float32，0-1范围）
            ld_img = f["LD"][idx]  # 低剂量CT（输入）
            nd_img = f["ND"][idx]  # 高剂量CT（标签）

        # 转换为Tensor并添加通道维：(H, W) → (1, H, W)
        ld_tensor = torch.from_numpy(ld_img).unsqueeze(0).float()
        nd_tensor = torch.from_numpy(nd_img).unsqueeze(0).float()

        # Resize到目标尺寸（适配神经网络输入）
        ld_tensor = self.transform(ld_tensor)
        nd_tensor = self.transform(nd_tensor)

        # -------------------------------
        # ✅ 数据增强（仅在训练集启用，与参考代码一致）
        # -------------------------------
        if self.augment:
            # 水平翻转
            if random.random() < 0.5:
                ld_tensor = torch.flip(ld_tensor, dims=[2])
                nd_tensor = torch.flip(nd_tensor, dims=[2])
            # 垂直翻转
            if random.random() < 0.5:
                ld_tensor = torch.flip(ld_tensor, dims=[1])
                nd_tensor = torch.flip(nd_tensor, dims=[1])
            # 随机小角度旋转
            if random.random() < 0.3:
                angle = random.choice([-5, -3, 3, 5])
                ld_tensor = T.functional.rotate(ld_tensor, angle)
                nd_tensor = T.functional.rotate(nd_tensor, angle)

        # ------------------------------------
        # ✅ 像素标准化（确保0-1范围，HDF5已归一化，此处为双重保险）
        # ------------------------------------
        ld_tensor = ld_tensor.clamp(0.0, 1.0)
        nd_tensor = nd_tensor.clamp(0.0, 1.0)

        return ld_tensor, nd_tensor  # (输入: LD, 标签: ND)


# ------------------- 数据集加载辅助函数（创建Train/Val/Test Dataset） -------------------
def get_pair_list(data_dir, target_size=256):
    """
    快速创建训练集、验证集、测试集的Dataset实例
    :param data_dir: HDF5数据集文件夹路径（如./ND_LD_HDF5_Dataset_0.7）
    :param target_size: 图像目标尺寸（默认256）
    :return: train_dataset, val_dataset, test_dataset
    """
    # 拼接各数据集的HDF5文件路径
    train_hdf5 = os.path.join(data_dir, "train_dataset.h5")
    val_hdf5 = os.path.join(data_dir, "val_dataset.h5")
    

    # 创建Dataset实例（训练集启用增强，验证集/测试集禁用）
    train_dataset = CTDataset(
        hdf5_path=train_hdf5,
        target_size=target_size,
        augment=True
    )
    val_dataset = CTDataset(
        hdf5_path=val_hdf5,
        target_size=target_size,
        augment=False
    )
    

    # 打印数据集信息
    print(f"数据集加载完成：")
    print(f"训练集：{len(train_dataset)} 对图像")
    print(f"验证集：{len(val_dataset)} 对图像")
    print(f"目标图像尺寸：{target_size}×{target_size}")

    return train_dataset, val_dataset

