import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y
class top_branch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(top_branch, self).__init__()
        self.conv3x1 = nn.Conv2d(in_channels, out_channels, (3, 1), 1, (1, 0))
        self.conv1x3 = nn.Conv2d(out_channels, out_channels, (1, 3), 1, (0, 1))
        self.maxpooling = nn.MaxPool2d(1)
    def forward(self, x):
        conv3x1 = self.conv3x1(x)
        conv1x3 = self.conv1x3(conv3x1)
        maxpooling = self.maxpooling(conv1x3)
        return maxpooling
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            depthwise_conv(hidden_dim, hidden_dim, 3, 1, relu=False) if stride == 2 else nn.Sequential(),
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )
        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
class ThreeBranchConv(nn.Module):
    def __init__(self, inp, oup, mid_channels=None):
        super(ThreeBranchConv, self).__init__()
        self.branch_top = top_branch(inp, oup)
        self.branch_middle = GhostModule(inp, oup)
        self.branch_down = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(1))
    def forward(self, x):
        return self.branch_top(x) + self.branch_middle(x) + self.branch_down(x)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
class Downsample_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ThreeBranchConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ThreeBranchConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Downsample_block(64, 128)
        self.down2 = Downsample_block(128, 256)
        self.down3 = Downsample_block(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Downsample_block(512, 1024 // factor)
        self.up1 = Upsample_block(1024, 512 // factor, bilinear)
        self.up2 = Upsample_block(512, 256 // factor, bilinear)
        self.up3 = Upsample_block(256, 128 // factor, bilinear)
        self.up4 = Upsample_block(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
