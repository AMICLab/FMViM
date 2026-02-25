import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff

def pixel_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)
    target_class = torch.argmax(targets, dim=1)
    correct = (preds == target_class).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()


def dice_score(outputs, targets, smooth=1e-5):
    num_classes = outputs.shape[1]
    preds = torch.argmax(outputs, dim=1)
    targets = torch.argmax(targets, dim=1)

    dices = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        intersection = (pred_cls * tgt_cls).sum()
        dice = (2 * intersection + smooth) / (pred_cls.sum() + tgt_cls.sum() + smooth)
        dices.append(dice.item())

    return np.mean(dices)


def iou_score(outputs, targets, smooth=1e-5):
    num_classes = outputs.shape[1]
    preds = torch.argmax(outputs, dim=1)
    targets = torch.argmax(targets, dim=1)

    ious = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        intersection = (pred_cls * tgt_cls).sum()
        union = pred_cls.sum() + tgt_cls.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou.item())

    return np.mean(ious)


def mean_iou(outputs, targets):
    return iou_score(outputs, targets)


# -------------------------
# 新增：Precision、Sensitivity、Jaccard（其实就是 IoU）
# -------------------------
def precision_score(outputs, targets, smooth=1e-5):
    num_classes = outputs.shape[1]
    preds = torch.argmax(outputs, dim=1)
    targets = torch.argmax(targets, dim=1)

    precisions = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        TP = (pred_cls * tgt_cls).sum()
        FP = (pred_cls * (1 - tgt_cls)).sum()

        precision = (TP + smooth) / (TP + FP + smooth)
        precisions.append(precision.item())

    return np.mean(precisions)


def sensitivity_score(outputs, targets, smooth=1e-5):
    """Sensitivity = Recall"""
    num_classes = outputs.shape[1]
    preds = torch.argmax(outputs, dim=1)
    targets = torch.argmax(targets, dim=1)

    recalls = []
    for cls in range(1, num_classes):
        pred_cls = (preds == cls).float()
        tgt_cls = (targets == cls).float()

        TP = (pred_cls * tgt_cls).sum()
        FN = ((1 - pred_cls) * tgt_cls).sum()

        recall = (TP + smooth) / (TP + FN + smooth)
        recalls.append(recall.item())

    return np.mean(recalls)


def jaccard_score(outputs, targets, smooth=1e-5):
    """Jaccard = IoU"""
    return iou_score(outputs, targets, smooth=smooth)


# -------------------------
# Hausdorff
# -------------------------
def hausdorff_distance(mask1, mask2):
    pts1 = np.argwhere(mask1 > 0)
    pts2 = np.argwhere(mask2 > 0)

    if pts1.size == 0 or pts2.size == 0:
        return 0.0

    return max(
        directed_hausdorff(pts1, pts2)[0],
        directed_hausdorff(pts2, pts1)[0]
    )


# -------------------------
# 综合指标
# -------------------------
def compute_all_metrics(outputs, targets):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    gt = torch.argmax(targets, dim=1).cpu().numpy()

    # Hausdorff：只计算前景
    hd_list = []
    for cls in np.unique(gt):
        if cls == 0:
            continue
        hd = hausdorff_distance(preds == cls, gt == cls)
        hd_list.append(hd)

    return {
        "acc": pixel_accuracy(outputs, targets),
        "dice": dice_score(outputs, targets),
        "iou": iou_score(outputs, targets),
        "miou": mean_iou(outputs, targets),
        "jaccard": jaccard_score(outputs, targets),        # 新增
        "sensitivity": sensitivity_score(outputs, targets), # 新增
        "precision": precision_score(outputs, targets),     # 新增
        "hausdorff": float(np.mean(hd_list)) if len(hd_list) else 0.0,
    }
