from torch import nn
import torch

class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, x, y):
        raise NotImplementedError()


class L2Loss(DistanceLoss):
    def forward(self, x, y):
        return (x - y).pow(2).sum(1).pow(0.5)
