import torch
from torch import nn

# Matthew weight L2 loss
class MWL2(nn.Module):
    def __init__(self, k, b):
        super(MWL2, self).__init__()
        self.k = k
        self.b = b
        self.L2 = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        I = torch.where(target >= 0, target + 1, target - 1)
        w = self.k * target + self.b * I

        loss = self.L2(pred, target)
        loss = w * loss
        loss = loss.mean()
        return loss