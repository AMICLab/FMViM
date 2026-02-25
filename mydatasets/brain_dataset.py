import os
import json
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

import torch

from mydatasets.transforms import NormalizePETto255

class BrainDataset(Dataset):
    def __init__(self, json_file, sam_class_stats_file, gt_class_stats_file, transform=None, merge_prior=True):
        """
        Args:
            json_file (str): train_samples.json 或 test_samples.json
            sam_class_stats_file (str): 包含 all_classes 的 sam_class_stats.json
            gt_class_stats_file (str): 包含 all_classes 的 GT_class_stats.json
            transform (callable, optional): 数据增强/预处理
            merge_prior (bool): 是否将多通道 SAM 先验融合为单通道
        """
        with open(json_file, "r") as f:
            self.samples = json.load(f)

        # SAM 类别映射
        with open(sam_class_stats_file, "r") as f:
            stats = json.load(f)
        self.sam_classes = sorted([int(c) for c in stats["all_classes"].keys()])
        self.num_sam_classes = len(self.sam_classes)
        self.sam_class2channel = {cls: idx for idx, cls in enumerate(self.sam_classes)}

        # GT 类别映射（含背景0）
        with open(gt_class_stats_file, "r") as f:
            gt_stats = json.load(f)
        self.gt_classes = sorted([int(c) for c in gt_stats["all_classes"].keys()])
        self.num_gt_classes = len(self.gt_classes) + 1
        self.gt_class2channel = {cls: idx + 1 for idx, cls in enumerate(self.gt_classes)}
        self.gt_class2channel[0] = 0

        self.transform = transform
        self.merge_prior = merge_prior

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample["h5"], "r") as f:
            h5 = f[list(f.keys())[0]][()]
        # -------------------
        # 1. 读取 T1 和 PET
        # -------------------
        t1 = h5[0]
        pet = h5[1]

        # -------------------
        # 2. 读取 GT (one-hot)
        # -------------------
        gt_arr = h5[2]
        h, w = gt_arr.shape
        gt = np.zeros((self.num_gt_classes, h, w), dtype=np.float32)
        unique_classes = np.unique(gt_arr)
        for cls_id in unique_classes:
            if cls_id not in self.gt_class2channel:
                continue
            ch = self.gt_class2channel[cls_id]
            gt[ch] = (gt_arr == cls_id).astype(np.float32)

        # -------------------
        # 3. 读取 SAM prior
        # -------------------
        sam_prior = h5[4]

        # -------------------
        # 5. 转为 Tensor
        # -------------------
        t1 = torch.from_numpy(np.expand_dims(t1, axis=0))
        pet = torch.from_numpy(np.expand_dims(pet, axis=0))
        sam_prior = torch.from_numpy(np.expand_dims(sam_prior, axis=0))
        gt = torch.from_numpy(gt)

        # -------------------
        # 6. 数据增强（可选）
        # -------------------
        if self.transform:
            sample_name = sample["h5"].split("/")[-3]
            t1, pet, sam_prior, gt = self.transform(t1, pet, sam_prior, gt,sample_name)

        return {
            "t1": t1,               # [1, H, W]
            "pet": pet,             # [1, H, W]
            "sam_prior": sam_prior, # [1, H, W]
            "gt": gt,               # [C_gt, H, W]
            "meta": sample
        }


if __name__ == "__main__":
    transform = NormalizePETto255()
    dataset = BrainDataset(
        "./data/test_samples.json",
        "./data/GT_class_stats_dlmuse.json",
        "./data/GT_class_stats_mask.json",
        transform=transform,
    )

    print("样本数:", len(dataset))
    data = dataset[1050]
    print("T1 shape:", data["t1"].shape)
    print("PET shape:", data["pet"].shape)
    print("SAM prior shape:", data["sam_prior"].shape)
    print("GT shape:", data["gt"].shape)
    print("Meta:", data["meta"])
