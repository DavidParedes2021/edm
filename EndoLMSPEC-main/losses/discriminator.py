"""Discriminator for Adversarial Loss"""
from unet.unet_parts import *


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # d: discriminator 
        self.d_1 = ConvLReLU(3, 8)
        self.d_2 = ConvLReLU(8, 16, batch_norm=True)
        self.d_3 = ConvLReLU(16, 32)
        self.d_4 = ConvLReLU(32, 64)
        self.d_5 = ConvLReLU(64, 128)
        self.d_6 = ConvLReLU(128, 128)
        self.d_7 = ConvLReLU(128, 256)
        self.d_out = DiscOut(256, 1)
    
    def forward(self, y_hat):
        if y_hat.shape[2] != 256 and y_hat.shape[3] != 256:
            y_hat = F.interpolate(y_hat, (256, 256), mode='bilinear', align_corners=True)
        y1 = self.d_1(y_hat)
        y2 = self.d_2(y1)
        y3 = self.d_3(y2)
        y4 = self.d_4(y3)
        y5 = self.d_5(y4)
        y6 = self.d_6(y5)
        y7 = self.d_7(y6)
        y_dout = self.d_out(y7)
        
        return y_dout  # it outputs a scalar for each image from the batch.