import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
import h5py
import json
from collections import defaultdict

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
# 验证循环（修改版）
# ------------------------------------------
def validate(model, loader, loss_fn, device, save_dir):
    model.eval()

    # 每个样本结果
    sample_results = []

    # 统计总指标
    total = defaultdict(float)
    count = 0

    # h5 文件存预测输出
    h5_path = os.path.join(save_dir, "outputs.h5")
    h5_file = h5py.File(h5_path, "w")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            t1 = batch["t1"].to(device)
            pet = batch["pet"].to(device)
            prior = batch["sam_prior"].to(device)
            gt = batch["gt"].to(device)
            meta = batch["meta"]  # 关键字段（list）

            outputs = model(t1,pet,prior)
            loss = loss_fn(outputs, gt)
            metrics = compute_all_metrics(outputs, gt)

            B = t1.shape[0]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for b in range(B):
                sample_name = meta['h5'][b]

                # 保存预测 mask 到 H5
                grp = h5_file.create_group(sample_name)
                grp.create_dataset("pred", data=preds[b], compression="gzip")

                # 保存指标
                result_dict = {
                    "meta": sample_name,
                    "loss": float(loss.item()),
                }
                result_dict.update({k: float(metrics[k]) for k in metrics})
                sample_results.append(result_dict)

                # 累加
                for k in metrics:
                    total[k] += float(metrics[k])
                total["loss"] += float(loss.item())
                count += 1

    h5_file.close()

    # 计算平均
    avg_results = {k: total[k] / count for k in total}

    return sample_results, avg_results


# ------------------------------------------
# 训练主函数
# ------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = NormalizePETto255()
    model_name = "our_big"

    # 保存目录
    save_dir = f"./trains/eval_result/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    val_set = BrainDataset(
        "./data/val_samples_clean.json",
        "./data/GT_class_stats_dlmuse.json",
        "./data/GT_class_stats_mask.json",
        transform=transform,
    )

    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4)

    model = CCViMFusionVNet().to(device)

    loss_fn = SegmentationLoss()

    pretrained_path = "./trains/checkpoints/CCViMFusionVNet_big/epoch_30.pth"
    if os.path.exists(pretrained_path):
        print(f"✅ Loading pretrained model from: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=device))

        sample_results, avg_results = validate(model, val_loader, loss_fn, device, save_dir)

    # 保存 sample-wise 结果
    json.dump(sample_results, open(os.path.join(save_dir, "eval_samples.json"), "w"), indent=2)

    # 保存平均结果
    json.dump(avg_results, open(os.path.join(save_dir, "eval_avg.json"), "w"), indent=2)

    print("\n===== ✅ Evaluation Finished =====")
    print(f"📌 Sample results: {save_dir}/eval_samples.json")
    print(f"📌 Average results: {save_dir}/eval_avg.json")
    print(f"📌 Predictions saved: {save_dir}/outputs.h5\n")

    print(
        f"Avg Loss: {avg_results['loss']:.4f} | "
        f"Dice: {avg_results['dice']:.4f} | "
        f"IoU: {avg_results['iou']:.4f} | "
        f"mIoU: {avg_results['miou']:.4f} | "
        f"HD: {avg_results['hausdorff']:.1f}"
    )


if __name__ == "__main__":
    main()
