import torch
import torch.nn as nn
import torch.nn.functional as F



class SharedDropout(nn.Module):
    def __init__(self):
        super(SharedDropout, self).__init__()
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, x):
        if self.training:
            assert self.mask is not None
            out = x * self.mask
            return out
        else:
            return x
