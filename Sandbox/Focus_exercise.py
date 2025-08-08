
# Focus Layer!

import torch
from torch import nn

class Focus(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c * 4, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(torch.cat([
            X[..., ::2, ::2],
            X[..., 1::2, ::2],
            X[..., ::2, 1::2],
            X[..., 1::2, 1::2]
        ], dim=1))