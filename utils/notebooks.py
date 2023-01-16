import matplotlib.pyplot as plt

def show(img, layer=None):
    if img.requires_grad:
        img = img.detach()
    if layer is not None:
        plt.imshow(img[0][layer], cmap="gray")
        plt.show()
    else:
        plt.imshow(img[0], cmap="gray")
        plt.show()