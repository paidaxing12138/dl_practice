import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from loguru import logger
from torch.utils.data import DataLoader


def get_dataloader(batch_size=32, image_size=224):
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 缩放到 32*32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化到 [-1, 1]
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 缩放到 32*32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化到 [-1, 1]
    ])
    # 加载 CIFAR-10 数据
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    return train_loader, val_loader