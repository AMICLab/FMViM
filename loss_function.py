import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        inputs = inputs.contiguous().view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.contiguous().view(targets.shape[0], targets.shape[1], -1)

        intersection = (inputs * targets).sum(dim=2)
        dice = (2. * intersection + self.smooth) / (inputs.sum(dim=2) + targets.sum(dim=2) + self.smooth)

        if self.ignore_background:
            dice = dice[:, 1:]  # 去掉通道0（背景）

        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] logits
        targets: [B, H, W] or [B, C, H, W] one-hot
        """
        # 允许 one-hot mask
        if targets.dim() == 4:
            targets = torch.argmax(targets, dim=1)  # [B, H, W]
        # Softmax -> prob
        probs = torch.softmax(inputs, dim=1)         # [B, C, H, W]
        # 取出目标类别概率 p_t
        pt = probs.gather(1, targets.unsqueeze(1))   # [B, 1, H, W]
        pt = pt.squeeze(1)
        # focal loss
        focal = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-6)
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal

class SegmentationLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(SegmentationLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] (logits)
        targets: [B, C, H, W] (one-hot mask)
        """
        B, C, H, W = inputs.shape
        _, C_gt, _, _ = targets.shape

        if C_gt == 1:
            # 二分类 (targets 是单通道 0/1)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float())
        else:
            # 多分类
            target_class = torch.argmax(targets, dim=1)  # [B, H, W]
            ce_loss = self.ce(inputs, target_class)
            focal_loss = self.focal(inputs, target_class)

        dice_loss = DiceLoss()(inputs, targets)

        return self.weight_ce * ce_loss + self.weight_dice * dice_loss + focal_loss
