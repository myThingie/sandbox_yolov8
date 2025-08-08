class Focus(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c * 4, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )

    def forward(self, x):
        # Slice and concat like YOLOv5/8 Focus
        return self.conv(torch.cat([
            x[..., ::2, ::2],    # top-left
            x[..., 1::2, ::2],   # top-right
            x[..., ::2, 1::2],   # bottom-left
            x[..., 1::2, 1::2]   # bottom-right
        ], dim=1))
