import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torch

def show(img: torch.Tensor, layer=None):
    img = img.cpu()
    img = F.convert_image_dtype(img, torch.float32)
    if img.requires_grad:
        img = img.detach()
    if layer is not None:
        plt.imshow(img[0][layer], cmap="gray")
        plt.show()
    else:
        plt.imshow(img[0], cmap="gray")
        plt.show()