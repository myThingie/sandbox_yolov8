class YOLOv8Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 64, 3, 1, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.csp1 = CSPBlock(128, 128, n=3)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.csp2 = CSPBlock(256, 256, bias=False, n=6)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        self.csp3 = CSPBlock(512, 512, n=3)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.csp1(x)
        x = self.conv2(x)
        x = self.csp2(x)
        x = self.conv3(x)
        x = self.csp3(x)
        return x