""" Full assembly of the parts to form the complete network """
from .unet_parts import *


class UNet24(nn.Module):
    def __init__(self, n_channels, res_layer=False, bilinear=False):
        super(UNet24, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.res_layer = res_layer

        """For first encoder-decoders (Level 4, 3 and 2)"""
        self.inc = DoubleConv(n_channels, 24)
        self.down1 = Down(24, 48)
        self.down2 = Down(48, 96)
        self.down3 = Down(96, 192)
        factor = 2 if bilinear else 1
        self.down4 = Down(192, 384 // factor)
        self.up1 = Up(384, 192 // factor, bilinear)
        self.up2 = Up(192, 96 // factor, bilinear)
        self.up3 = Up(96, 48 // factor, bilinear)
        self.up4 = Up(48, 24, bilinear)
        self.outc = OutConv(24, n_channels)  # Here we have to compute 3 channels instead of 3 classes.
        self.out_up = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2) #

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        
        if self.res_layer:
            output = self.outc(x9)
            output = torch.add(output, x)
        else:
            output = self.outc(x9)
        
        output = self.out_up(output)
        return output


class UNet16(nn.Module):
    def __init__(self, n_channels, res_layer=False, bilinear=False):
        super(UNet16, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.res_layer = res_layer

        """For last encoder-decoder (Level1)"""
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)  # for 128 this result is 8x8x256
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_channels)   # Here we have to compute 3 channels instead of 3 classes.
        

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        
        if self.res_layer:
            output = self.outc(x9)
            output = torch.add(output, x)
        else:
            output = self.outc(x9)
        
        return output