import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None):
    super(DoubleConv, self).__init__()
    if not mid_channels:
      mid_channels = out_channels

    self.double_conv = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.double_conv(x)

class Down(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()

    self.maxpool_conv = nn.Sequential(
        nn.MaxPool2d(kernel_size=2, stride=2),
        DoubleConv(in_channels, out_channels)
    )

  def forward(self, x):
    return self.maxpool_conv(x)

class Up(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(Up, self).__init__()

    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x)
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    diffX_dv2 = torch.div(diffX, 2, rounding_mode='floor')
    diffY_dv2 = torch.div(diffY, 2, rounding_mode='floor')

    x1 = F.pad(x1, [diffX_dv2, diffX - diffX_dv2, diffY_dv2, diffY - diffY_dv2])
    x = torch.cat([x2, x1], dim=1)
    return self.conv(x)

class OutCv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(OutCv,self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)

class Unet(torch.nn.Module):
  def __init__(self, n_channels, n_classes, base_factor=32):
    super(Unet, self).__init__()

    self.n_channels = n_channels
    self.n_classes = n_classes
    self.base_factor = base_factor

    # Conracting Path
    self.inc = DoubleConv(n_channels, base_factor)
    self.down1 = Down(base_factor, base_factor * 2)
    self.down2 = Down(base_factor * 2, base_factor * 4)
    self.down3 = Down(base_factor * 4, base_factor * 8)
    self.down4 = Down(base_factor * 8, base_factor * 16)


    # Expansive Path
    self.up1 = Up(base_factor * 16, base_factor * 8)
    self.up2 = Up(base_factor * 8, base_factor * 4)
    self.up3 = Up(base_factor * 4, base_factor * 2)
    self.up4 = Up(base_factor * 2, base_factor)

    self.outc = OutCv(base_factor, n_classes)

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
