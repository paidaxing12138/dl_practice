import os
import torch as t
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.cnn import MyCNN
from dataset.cifar10 import get_dataloader
from loguru import logger


def train(model, device, train_loader, criterion, optimizer, num_epochs):
    model = model.to(device)
    best_loss = float('inf')
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs), ncols=100):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # logger.debug(f"images: {images.shape} labels: {labels.shape}")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.shape[0]
            # calculate precision
            _, predicted = t.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss /= len(train_loader.dataset)
        acc = correct / total
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            t.save(model.state_dict(), f"save/best_loss_epoch_{epoch+1}.pth")
        if acc > best_acc:
            best_acc = acc
            t.save(model.state_dict(), f"save/best_acc_epoch_{epoch+1}.pth")
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {acc:.4f}")
                   

if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # print("CUDA Available:", t.cuda.is_available())
    # print("CUDA Version:", t.version.cuda)
    # print("cuDNN Enabled:", t.backends.cudnn.enabled)
    # print("cuDNN Version:", t.backends.cudnn.version())
    # print("Number of GPUs:", t.cuda.device_count())
    # print(t.__version__)
    # print(t.version.cuda)
    t.backends.cudnn.enabled = False
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    input_size = 224
    train_loader, val_loader = get_dataloader(batch_size=batch_size, image_size=input_size)
    num_classes = len(train_loader.dataset.classes)
    model = MyCNN(num_class=num_classes, input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train(model, device, train_loader, criterion, optimizer, num_epochs)
