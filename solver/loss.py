import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable


# https://amaarora.github.io/2020/06/29/FocalLoss.html
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.01, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, preds, targets):
        BCE_loss = F.binary_cross_entropy(preds.view(-1), targets.view(-1).float(), reduction='none')
        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(preds.device)
        
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = F_loss.mean()

        if math.isnan(F_loss) or math.isinf(F_loss):
            F_loss = torch.zeros(1).to(preds.device)

        return F_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # outputs = torch.sigmoid(preds.squeeze())

        numerator = 2 * torch.sum(preds * targets) + self.smooth
        denominator = torch.sum(preds ** 2) + torch.sum(targets ** 2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return soft_dice_loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.alpha = 0.4
        self.dice_loss = DiceLoss()
        self.focal_loss = WeightedFocalLoss()
        self.w_dice = 1.
        self.w_focal = 100.
        self.S_dice = []
        self.S_focal = []

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds.squeeze())
        
        dice_loss = self.dice_loss(preds, targets) * self.w_dice
        focal_loss = self.focal_loss(preds, targets) * self.w_focal

        return dice_loss, focal_loss