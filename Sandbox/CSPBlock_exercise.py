
# CSP Block. CSP: Cross Stage Partial Block

import torch
from torch import nn

class CSPBlock(nn.Module):
    def __init(self, in_c, out_c, n=1):
        super().__init__()
        mid_c = outc // 2

        self.split = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1, 1, 0),
            nn.BatchNorm2d(mid_c),
            nn.SiLU()
        )

        def conv_block():
            return nn.Sequential(
                nn.Conv2d(mid_c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(mid_c),
                nn.SiLU()
            )

        self.blocks = nn.Sequential(
            *[conv_block() for _ in range(n)]
            )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            ss.SiLU()
        )


    def frorward(self, x):
        x1 = slef.split(x)
        x2 = self.blocks(x1)
        return self.fuse(torch.cat([x1, x2], dim=1))

