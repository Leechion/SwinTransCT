<<<<<<< HEAD
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import random


class CTDataset(Dataset):
    """CT数据集加载器：增强 + 标准化 + 更快加载"""

    def __init__(self, pair_list, target_size=256, augment=True):
        self.samples = pair_list
        self.target_size = target_size
        self.augment = augment

        # 预定义 torchvision 变换管道（更快更稳定）
        self.to_tensor = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # 自动转 [C,H,W], 并除以255
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ld_path, nd_path = self.samples[idx]

        # 读取
        ld_img = Image.open(ld_path).convert('L')
        nd_img = Image.open(nd_path).convert('L')

        # Resize + 转Tensor
        ld_tensor = self.to_tensor(ld_img)
        nd_tensor = self.to_tensor(nd_img)

        # -------------------------------
        # ✅ 数据增强（仅在训练集启用）
        # -------------------------------
        if self.augment:
            if random.random() < 0.5:
                ld_tensor = torch.flip(ld_tensor, dims=[2])  # 水平翻转
                nd_tensor = torch.flip(nd_tensor, dims=[2])
            if random.random() < 0.5:
                ld_tensor = torch.flip(ld_tensor, dims=[1])  # 垂直翻转
                nd_tensor = torch.flip(nd_tensor, dims=[1])
            if random.random() < 0.3:
                angle = random.choice([-5, -3, 3, 5])
                ld_tensor = T.functional.rotate(ld_tensor, angle)
                nd_tensor = T.functional.rotate(nd_tensor, angle)

        # ------------------------------------
        # ✅ 像素标准化（将像素归一化到 [0,1]）
        # ------------------------------------
        ld_tensor = ld_tensor.clamp(0.0, 1.0)
        nd_tensor = nd_tensor.clamp(0.0, 1.0)

        return ld_tensor, nd_tensor


# ------------------- 数据集加载函数 -------------------
def get_pair_list(data_dir, split='train'):
    ld_dir = os.path.join(data_dir, split, 'LD')
    nd_dir = os.path.join(data_dir, split, 'ND')

    ld_files = sorted([f for f in os.listdir(ld_dir) if f.lower().endswith('.png')])
    nd_files = sorted([f for f in os.listdir(nd_dir) if f.lower().endswith('.png')])

    pair_list = [
        (os.path.join(ld_dir, ld_f), os.path.join(nd_dir, nd_f))
        for ld_f, nd_f in zip(ld_files, nd_files)
    ]
    return pair_list
