import sys
import os
# 把项目根目录加入 Python 模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from models.CCViM_fusion_vnet import CCViMFusionVNet
from mydatasets.brain_dataset import BrainDataset
from loss_function import SegmentationLoss
from metrics import compute_all_metrics
import matplotlib.colors as mcolors
import numpy as np
import os
from mydatasets.transforms import NormalizePETto255

try:
    import colorcet as cc
    has_cc = True
except ImportError:
    has_cc = False


def generate_distinct_colors(n):
    """生成 n 种高区分度颜色（背景为黑色）"""
    if has_cc and n > 1:
        base_colors = cc.glasbey[:n - 1]
    else:
        base_colors = plt.cm.hsv(np.linspace(0, 1, n - 1))
    return ["black"] + list(base_colors)

# ------------------------------------------
# 可视化预测函数
# ------------------------------------------
torch.no_grad()
def visualize_prediction(model, dataset, device, epoch, save_dir,
                         color_map_path="./data/class_color_map_0_200.json"):
    """
    随机选择一张样本，保存 T1 + PET + Prior + Pred + GT 对比图
    使用外部JSON颜色映射文件（0固定为黑色）
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # === 加载颜色映射表 ===
    with open(color_map_path, "r") as f:
        color_dict = json.load(f)

    # 将字符串hex转为matplotlib可用的RGB列表
    max_class = max(map(int, color_dict.keys()))
    color_list = [color_dict[str(i)] if str(i) in color_dict else "#000000" for i in range(max_class + 1)]
    cmap_fixed = mcolors.ListedColormap(color_list)
    norm_fixed = mcolors.BoundaryNorm(np.arange(max_class + 2) - 0.5, cmap_fixed.N)

    # === 随机抽样 ===
    idx = torch.randint(0, len(dataset), (1,)).item()
    sample = dataset[idx]

    # === 数据准备 ===
    t1 = sample["t1"].unsqueeze(0).to(device)
    pet = sample["pet"].unsqueeze(0).to(device)
    prior = sample["sam_prior"].unsqueeze(0).to(device)
    gt = sample["gt"].squeeze().cpu().numpy()

    # === 模型预测 ===
    output = model(t1, pet, prior)
    output = torch.softmax(output, dim=1).detach().squeeze().cpu().numpy()

    # === GT / PRIOR / PRED 转为类别图 ===
    gt_mask = np.argmax(gt, axis=0)
    prior_mask = prior.squeeze().cpu().numpy()
    pred_mask = np.argmax(output, axis=0)

    # === 原始T1、PET影像 ===
    t1_img = t1.squeeze().cpu().numpy()
    if t1_img.ndim == 3:
        t1_img = np.mean(t1_img, axis=0)

    pet_img = pet.squeeze().cpu().numpy()
    if pet_img.ndim == 3:
        pet_img = np.mean(pet_img, axis=0)

    # === 屏蔽背景 ===
    masked_prior = np.ma.masked_where(prior_mask == 0, prior_mask)
    masked_pred = np.ma.masked_where(pred_mask == 0, pred_mask)
    masked_gt = np.ma.masked_where(gt_mask == 0, gt_mask)

    # === 绘图：2行3列 ===
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))

    # 第一行：原始输入
    axs[0, 0].imshow(t1_img, cmap="gray", origin="lower")
    axs[0, 0].set_title(f"T1 Image (idx={idx})")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(pet_img, cmap="gray", origin="lower")
    axs[0, 1].set_title("PET Image")
    axs[0, 1].axis("off")

    axs[0, 2].axis("off")

    # 第二行：Prior / Pred / GT
    axs[1, 0].imshow(t1_img, cmap="gray", origin="lower")
    im_prior = axs[1, 0].imshow(masked_prior, cmap=cmap_fixed, norm=norm_fixed, alpha=0.5, origin="lower")
    axs[1, 0].set_title("T1 + Prior")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(t1_img, cmap="gray", origin="lower")
    im_pred = axs[1, 1].imshow(masked_pred, cmap=cmap_fixed, norm=norm_fixed, alpha=0.5, origin="lower")
    axs[1, 1].set_title("T1 + Predicted Segmentation")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(t1_img, cmap="gray", origin="lower")
    im_gt = axs[1, 2].imshow(masked_gt, cmap=cmap_fixed, norm=norm_fixed, alpha=0.5, origin="lower")
    axs[1, 2].set_title("Ground Truth Mask")
    axs[1, 2].axis("off")

    # === 添加颜色条（只显示一次即可） ===
    cbar = plt.colorbar(im_gt, ax=axs[1, 2], fraction=0.046, pad=0.04)
    cbar.set_label("Class ID", rotation=270, labelpad=10)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch}_sample_{idx}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[Visualize] Epoch {epoch} sample {idx} saved to {save_path}")




# ------------------------------------------
# 训练与验证循环
# ------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = {"loss": 0, "acc": 0, "dice": 0, "iou": 0, "miou": 0, "hausdorff": 0, "jaccard": 0,"sensitivity":0,"precision":0}

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        t1 = batch["t1"].to(device)
        pet = batch["pet"].to(device)
        prior = batch["sam_prior"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad()
        outputs = model(t1, pet, prior)
        loss = loss_fn(outputs, gt)
        loss.backward()
        optimizer.step()

        metrics = compute_all_metrics(outputs, gt)

        total["loss"] += loss.item()
        for k in metrics:
            total[k] += metrics[k]

        pbar.set_postfix({
            "loss": f"{loss.item():.2f}",
            "dice": f"{metrics['dice']:.2f}",
            "iou": f"{metrics['iou']:.2f}",
            "miou": f"{metrics['miou']:.2f}",
            "hd": f"{metrics['hausdorff']:.1f}"
        })

    n = len(loader)
    return {k: total[k] / n for k in total}



def validate(model, loader, loss_fn, device):
    model.eval()
    total = {"loss": 0, "acc": 0, "dice": 0, "iou": 0, "miou": 0, "hausdorff": 0, "jaccard": 0,"sensitivity":0,"precision":0}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            t1 = batch["t1"].to(device)
            pet = batch["pet"].to(device)
            prior = batch["sam_prior"].to(device)
            gt = batch["gt"].to(device)

            outputs = model(t1, pet, prior)
            loss = loss_fn(outputs, gt)
            metrics = compute_all_metrics(outputs, gt)

            total["loss"] += loss.item()
            for k in metrics:
                total[k] += metrics[k]

    n = len(loader)
    return {k: total[k] / n for k in total}



# ------------------------------------------
# 训练主函数
# ------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = NormalizePETto255()

    train_set = BrainDataset(
        "./data/train_samples.json",
        "./data/GT_class_stats_dlmuse.json",
        "./data/GT_class_stats_mask.json",
        transform=transform,
    )
    val_set = BrainDataset(
        "./data/val_samples_clean.json",
        "./data/GT_class_stats_dlmuse.json",
        "./data/GT_class_stats_mask.json",
        transform=transform,
    )
    test_set = BrainDataset(
        "./data/test_samples.json",
        "./data/GT_class_stats_dlmuse.json",
        "./data/GT_class_stats_mask.json",
        transform=transform,
    )

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)

    model = CCViMFusionVNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = SegmentationLoss()

    # pretrained_path = "./trains/checkpoints/CCViMFusionVNet_final3/epoch_30.pth"
    # if os.path.exists(pretrained_path):
    #     print(f"✅ Loading pretrained model from: {pretrained_path}")
    #     model.load_state_dict(torch.load(pretrained_path, map_location=device))

    num_epochs = 30
    save_dir = "checkpoints/CCViMFusionVNet_final3"
    vis_dir = "plots/CCViMFusionVNet_final3"
    plot_dir = "plots/CCViMFusionVNet_final3"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 保存曲线数据
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []

    for epoch in range(num_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_stats = validate(model, val_loader, loss_fn, device)
        test_stats = validate(model, test_loader, loss_fn, device)

        train_losses.append(train_stats["loss"])
        val_losses.append(val_stats["loss"])
        train_dices.append(train_stats["dice"])
        val_dices.append(val_stats["dice"])

        print(f"[Epoch {epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_stats['loss']:.4f} | "
              f"Dice: {train_stats['dice']:.4f} | "
              f"IoU: {train_stats['iou']:.4f} | "
              f"mIoU: {train_stats['miou']:.4f} | "
              f"HD: {train_stats['hausdorff']:.1f} || "
              f"Val Loss: {val_stats['loss']:.4f} | "
              f"Dice: {val_stats['dice']:.4f} | "
              f"IoU: {val_stats['iou']:.4f} | "
              f"mIoU: {val_stats['miou']:.4f} | "
              f"HD: {val_stats['hausdorff']:.1f} || "
              f"Test Loss: {test_stats['loss']:.4f} | "
              f"Dice: {test_stats['dice']:.4f} | "
              f"IoU: {test_stats['iou']:.4f} | "
              f"mIoU: {test_stats['miou']:.4f} | "
              f"HD: {test_stats['hausdorff']:.1f}")


        # torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch+1}_{val_loss:.2f}.pth"))

        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch + 1}.pth"))
            visualize_prediction(model, val_set, device, epoch + 1, vis_dir)

        # 每10个epoch更新一次曲线
        if (epoch) % 1 == 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss Curve")

            plt.subplot(1, 2, 2)
            plt.plot(train_dices, label='Train Dice')
            plt.plot(val_dices, label='Val Dice')
            plt.xlabel("Epoch")
            plt.ylabel("Dice")
            plt.legend()
            plt.title("Dice Curve")

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "metrics_curve.png"))
            plt.close()

if __name__ == "__main__":
    main()
