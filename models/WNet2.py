from monai.networks.nets import UNet
import torch
from torch import nn

class WNet(nn.Module):
    def __init__(self, inputs = 1):
        super().__init__()

        self.unet1 = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
        )
        self.batch_norm = nn.BatchNorm3d(inputs)
        inputs += 1
        self.unet2 = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=(8, 16, 32, 64, 128),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
        )

        # init, shamelessly lifted from torchvision/models/resnet.py
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if torch.cuda.is_available():
            self.cuda()
    
    def forward(self, x):
        x1 = self.unet1(x)
        x1 = self.batch_norm(x1)
        new_data = torch.cat([x, torch.sigmoid(x1)], dim=1)
        x2 = self.unet2(new_data)
        return x2
