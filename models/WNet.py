import torch
import torch.nn as nn

# https://amaarora.github.io/2020/09/13/unet.html

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c: int, out_c: int, k_sz=3):
        super(ConvBlock, self).__init__()

        pad = (k_sz - 1) // 2

        block = []
        # TODO max pooling

        block.append(nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        block.append(nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(out_c))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        # TODO max pooling
        out = self.block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, layers: list[int] = [8, 16, 32]):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, layers: list[int] = [32, 16, 8]):
        super().__init__()
        self.layers = layers

        self.upconvs = []
        for i in range(len(layers)-1):
            self.upconvs.append(nn.ConvTranspose2d(layers[i], layers[i+1], 2, 2))
        self.enc_blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpsampleBlock, self).__init__()
        block = []
        block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
        block.append(nn.Conv2d(in_c, out_c, kernel_size=1))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class ConvBridgeBlock(torch.nn.Module):
    def __init__(self, channels, k_sz=3):
        super(ConvBridgeBlock, self).__init__()
        pad = (k_sz - 1) // 2
        block=[]

        block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UpConvBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, k_sz=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
        super(UpConvBlock, self).__init__()
        self.conv_bridge = conv_bridge

        self.up_layer = UpsampleBlock(in_c, out_c, up_mode=up_mode)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        out = self.conv_layer(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_ch = 1, layers=(8, 16, 32)):
        super().__init__()

        self.first = ConvBlock(in_c=in_ch, out_c=layers[1])
        self.encoder = Encoder(layers)
        ...

