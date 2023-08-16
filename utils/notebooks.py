import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import torch
from torch import nn
from monai.losses import DiceLoss

def show_slice(x: torch.Tensor, y: torch.Tensor = None, layer=45):
    x = x.cpu()
    x = F.convert_image_dtype(x, torch.float32)

    if y is not None:
        y = y.cpu()
        y = F.convert_image_dtype(y, torch.float32)
        figure, axis = plt.subplots(1, 2, figsize=(10, 30))
        axis[0].imshow(x[0][layer], cmap='gray')
        axis[0].set_title('Input')
        axis[1].imshow(y[0][layer], cmap='gray')
        axis[1].set_title('ground truth')
        plt.show()
    else:
        plt.imshow(x[0][layer], cmap='gray')

def show(x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, layer=45):
    x, pred, y = x.cpu(), pred.cpu(), y.cpu()
    x = F.convert_image_dtype(x, torch.float32)
    pred = F.convert_image_dtype(pred, torch.float32)
    y = F.convert_image_dtype(y, torch.float32)
    for i in range(len(x)):
        figure, axis = plt.subplots(1, 3, figsize=(10, 30))
        axis[0].imshow(x[i][0][layer], cmap='gray')
        axis[0].set_title('Input')
        axis[1].imshow(pred[i][0][layer], cmap='gray')
        axis[1].set_title('prediction')
        axis[2].imshow(y[i][0][layer], cmap='gray')
        axis[2].set_title('ground truth')
        plt.show()


def plot_predictions(dataloader, model, layer=45):
    with torch.no_grad():
        model.cpu()
        for x, y in iter(dataloader):
            pred = model(x)
            pred = torch.nn.Sigmoid()(pred)
            print(pred.shape)
            show(x, pred, y, layer)
    model.cuda()


def train(model, train_dl, validation_dl, optimizer, criterion, epochs, writer=None, model_name='wnet'):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    lowest = None
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        for data in iter(train_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(inputs)
            loss = criterion(output, labels)
            
            if lowest is None or (loss < lowest and loss < 0.2 and epoch % 10 == 0):
                lowest = loss
                torch.save(model.state_dict(
                ), f'./state_dicts/{model_name}/best/{model_name}_epoch_{epoch}_loss_{loss:.4f}.pk')

            loss.backward()
            optimizer.step()
        if epoch % 250 == 0:
            torch.save(model.state_dict(
            ), f'./state_dicts/{model_name}/{model_name}_epoch_{epoch}_loss_{loss:.4f}.pk')

        if writer:
            writer.add_scalar("Loss/train", loss, epoch)
            if epoch % 2 == 0:
                # validation
                val_loss = eval(model, validation_dl, criterion)
                writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.flush()
        print(loss)

def eval(model, dl, criterion):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    running_loss = 0
    batch_size = 0
    print("validating")
    with torch.no_grad():
        for data in iter(dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            val_output = model(inputs)
            val_loss = criterion(val_output, labels)

            running_loss += val_loss.item()
            batch_size += 1
    print(f'validation loss: {running_loss / batch_size}')
    return running_loss / batch_size
