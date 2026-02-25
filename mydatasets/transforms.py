# datasets/transforms.py
import json

import numpy as np
import random
import cv2
import torch

def random_flip(mri, pet, prior, label, p=0.5):
    if random.random() < p:
        # horizontal flip
        mri = np.fliplr(mri).copy()
        pet = np.fliplr(pet).copy()
        prior = np.flip(prior, axis=-1).copy() if prior.ndim==3 else np.fliplr(prior).copy()
        label = np.fliplr(label).copy()
    return mri, pet, prior, label

def random_rotate(mri, pet, prior, label, max_angle=10):
    angle = random.uniform(-max_angle, max_angle)
    H,W = mri.shape
    M = cv2.getRotationMatrix2D((W/2,H/2), angle, 1.0)
    mri = cv2.warpAffine(mri, M, (W,H), flags=cv2.INTER_LINEAR)
    pet = cv2.warpAffine(pet, M, (W,H), flags=cv2.INTER_LINEAR)
    if prior.ndim==3:
        prior = np.stack([cv2.warpAffine(prior[c], M, (W,H), flags=cv2.INTER_LINEAR) for c in range(prior.shape[0])], axis=0)
    else:
        prior = cv2.warpAffine(prior, M, (W,H), flags=cv2.INTER_LINEAR)
    label = cv2.warpAffine(label.astype('int32'), M, (W,H), flags=cv2.INTER_NEAREST)
    return mri, pet, prior, label

class ComposeTransforms:
    def __init__(self, do_flip=True, do_rotate=True):
        self.do_flip = do_flip
        self.do_rotate = do_rotate
    def __call__(self, mri, pet, prior, label):
        if self.do_flip:
            mri, pet, prior, label = random_flip(mri, pet, prior, label)
        if self.do_rotate:
            mri, pet, prior, label = random_rotate(mri, pet, prior, label)
        return mri, pet, prior, label

class NormalizePETto255:
    def __init__(self):
        pet_stats_path = "./data/pet_stats.json"
        with open(pet_stats_path) as f:
            self.pet_stats = json.load(f)
        pass

    def __call__(self, t1, pet, sam_prior, gt, sample_name):
        """
        输入：
            t1: [1, H, W] Tensor
            pet: [1, H, W] Tensor
            sam_prior: [1, H, W] Tensor
            gt: [C_gt, H, W] Tensor
        输出：
            归一化后的 (t1, pet, sam_prior, gt)
        """
        stats = self.pet_stats[sample_name]
        # petz-score归一化
        mean = stats["mean"]
        std = stats["std"]
        pet = (pet - mean) / (std + 1e-5)

        # pet_min = torch.min(pet)
        # pet_max = torch.max(pet)
        # if pet_max > pet_min:  # 防止除0
        #     pet = (pet - pet_min) / (pet_max - pet_min) * 255.0
        # else:
        #     pet = torch.zeros_like(pet)

        return t1, pet, sam_prior, gt