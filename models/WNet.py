import torch
import torch.nn as nn

# https://amaarora.github.io/2020/09/13/unet.html

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvBlock(torch.nn.Module):
    def __init__(self, in_c: int, out_c: int, k_sz=3):
        super(ConvBlock, self).__init__()

        pad = (k_sz - 1) // 2

        block = []
        # TODO Galdran et. al. uses max pooling here

        block.append(nn.Conv3d(in_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm3d(out_c))

        block.append(nn.Conv3d(out_c, out_c, kernel_size=k_sz, padding=pad))
        block.append(nn.ReLU())
        block.append(nn.BatchNorm3d(out_c))

        self.block = nn.Sequential(*block)

    @torch.autocast(device_type=get_device())
    def forward(self, x):
        # TODO max pooling
        out = self.block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, layers: list[int] = [8, 16, 32]):
        super().__init__()
        self.enc_blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        # Max pooling does not work with bfloat16 on CPU
        self.pool = nn.MaxPool3d(2)

    @torch.autocast(device_type=get_device())
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
        self.upconvs = nn.ModuleList([nn.ConvTranspose3d((layers[i], [layers[i+1]], 2, 2) for i in range(len(layers) - 1))])
        self.dec_blocks = nn.ModuleList([ConvBlock(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 

    @torch.autocast(device_type=get_device())
    def forward(self, x, encoder_ftrs):
        for i in range(len(self.layers)-1):
            x = self.upconvs[i](x)
        
        return x

class UNet(nn.Module):
    def __init__(self, in_ch = 1, layers=(8, 16, 32)):
        super().__init__()

        self.first = ConvBlock(in_c=in_ch, out_c=layers[1])
        self.encoder = Encoder(layers)
        self.decoder = Decoder(layers.reverse())
        ...

