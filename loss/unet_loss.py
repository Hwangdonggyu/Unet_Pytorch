import torch
import torch.nn as nn

# 왜 unet에서 제공하는 loss 안쓰는지?
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - ((2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth))
    

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_dice = self.dice(pred, target)
        return loss_bce + loss_dice