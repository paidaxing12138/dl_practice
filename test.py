import os
import torch as t
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.cnn import MyCNN
from dataset.cifar10 import get_dataloader
from loguru import logger


def validate(model, device, val_loader):
    model = model.to(device)
    correct, total = 0, 0
    model.eval()    
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        with t.no_grad():
            output = model(images)
        _, predicted = t.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc
                 

if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    pth_path = r"save\best_loss_epoch_18.pth"
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    input_size = 224
    train_loader, val_loader = get_dataloader(batch_size=batch_size, image_size=input_size)
    num_classes = len(train_loader.dataset.classes)
    model = MyCNN(num_class=num_classes, input_size=input_size)

    model.load_state_dict(t.load(pth_path))
    acc = validate(model=model, device=device, val_loader=val_loader)
    logger.info(acc)
