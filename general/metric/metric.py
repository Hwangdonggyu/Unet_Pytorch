import torch
import torch.nn as nn


def cal_dice_score(pred, target, smooth=1e-5):
  pred = torch.sigmoid(pred)
  pred_flat = pred.view(-1)
  target_flat = target.view(-1)
  intersection = (pred_flat * target_flat).sum()
  dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
  dice_score = dice_score.clamp(0, 1)
  return dice_score

def cal_pixel_accuracy(pred, target):
    pred = torch.sigmoid(pred)  
    pred = (pred > 0.5).float()        
    target = target.float()  

    correct = (pred == target).float() 
    acc = correct.sum() / correct.numel()
    return acc.item()

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()

    def forward(self, pred, target):
        dice_score = cal_dice_score(pred, target)
        pixel_acc = cal_pixel_accuracy(pred, target)
        return dice_score, pixel_acc