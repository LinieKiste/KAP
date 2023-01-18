#!/bin/env python3
import models.get_model
import torch
from utils.loader import DicomDataset2D
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC

BATCH_SIZE = 20
EPOCHS = 10

def train(model, optimizer, criterion):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'

    model.train()

    train_dataloader = DataLoader(DicomDataset2D("data/train.csv"), batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(DicomDataset2D("data/test.csv"), batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        print("EPOCH ", epoch)
        for data in tqdm(iter(train_dataloader)):
            inputs, labels = data

            optimizer.zero_grad()

            _, output = model(inputs)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            # evaluate model
            print('evaluating...')
            with torch.no_grad():
                auroc = AUROC(task='binary')

                for data in iter(test_dataloader):
                    inputs, target = data
                    _, preds = model(inputs)

                    res = auroc(preds, target)
                    print(res)
                    break

    """
    for epoch in range(EPOCHS):
        print("Epoch ", epoch)
        accuracy = 0
        num_of_steps = 0
        for i in tqdm(range(0, len(train_features), BATCH_SIZE)):
            batch_features = train_features[i:i + BATCH_SIZE] #.view(-1, 1, 50, 50)
            batch_labels = train_labels[i:i+BATCH_SIZE]

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model.to(device)(batch_features)

            loss = loss_function(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        accuracy = test(model, test_features, test_labels)
        _loss = round(loss.item(), 3)
        accuracies.append(accuracy)
        losses.append(_loss)

        print("Accuracy:", accuracy)
        print("Loss:", _loss)
        """

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.get_model.get_arch('wnet', in_c=1)
    # model = models.get_own()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, optimizer, criterion)


if __name__ == '__main__':
    main()

