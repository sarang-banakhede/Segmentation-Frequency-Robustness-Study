import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64,  128);           self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256);           self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512);           self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2); self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512,  256, 2, 2); self.dec3 = DoubleConv(512,  256)
        self.up2 = nn.ConvTranspose2d(256,  128, 2, 2); self.dec2 = DoubleConv(256,  128)
        self.up1 = nn.ConvTranspose2d(128,   64, 2, 2); self.dec1 = DoubleConv(128,   64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        x  = self.dec4(torch.cat([self.up4(b), e4], 1))
        x  = self.dec3(torch.cat([self.up3(x), e3], 1))
        x  = self.dec2(torch.cat([self.up2(x), e2], 1))
        x  = self.dec1(torch.cat([self.up1(x), e1], 1))
        return self.head(x)
