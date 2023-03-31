from .unet3d.model import UNet3D
import torch
from torch import nn

class WNet(nn.Module):
    def __init__(self, inputs = 1, f_maps = 8, num_levels = 4):
        super().__init__()

        self.unet1 = UNet3D(inputs, 1, f_maps=8, num_levels=4)
        inputs += 1
        self.unet2 = UNet3D(inputs, 1, f_maps=8, num_levels=4)

        if torch.cuda.is_available():
            self.cuda()
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        return x2
