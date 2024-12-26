import os
import sys
import datetime
import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model.cnn import MyCNN
from dataset.cifar10 import get_dataloader
from loguru import logger


def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, writer):
    model = model.to(device)
    best_loss = float('inf')
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs), ncols=100):
        model.train()
        train_epoch_loss = 0
        train_correct = 0
        train_total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # logger.debug(f"images: {images.shape} labels: {labels.shape}")
            optimizer.zero_grad()
            output = model(images)
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item() * images.shape[0]
            # calculate precision
            _, predicted = t.max(output.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        train_epoch_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        logger.info(f"epoch {epoch+1}: training loss: {train_epoch_loss}, accuracy: {train_acc}")
        model.eval()
        val_epoch_loss = 0
        val_correct = 0
        val_total = 0
        with t.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                val_loss = criterion(output, labels)
                val_epoch_loss += val_loss.item() * images.shape[0]
                _, predicted = t.max(output.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_epoch_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        logger.info(f"epoch {epoch+1}: validating loss: {val_epoch_loss}, validating accuracy: {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            t.save(model.state_dict(), f"save/best_val_acc_epoch_{epoch+1}.pth")
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {best_acc:.4f} Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/Train', train_epoch_loss, epoch+1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch+1)
        writer.add_scalar('Loss/Validation', val_epoch_loss, epoch+1)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch+1)
    writer.close()
                
if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"logs/{os.path.basename(__file__)}_{current_time}.log"
    print(log_file_name)
    logger.add(log_file_name, level="INFO", format="{time} | {level} | {message}")
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    input_size = 224
    logger.info(f"Device: {device}\nInput size: {input_size}\nnumEpochs: {num_epochs}\nbatch_size: {batch_size}\nlearning_rate: {learning_rate}")
    train_loader, val_loader = get_dataloader(batch_size=batch_size, image_size=input_size)
    num_classes = len(train_loader.dataset.classes)
    model = MyCNN(num_class=num_classes, input_size=input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    tensorboard_dir = os.path.join("tensorboard_log", current_time)
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs, writer)
