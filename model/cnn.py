import torch as t
import torch.nn as nn
from loguru import logger

class MyCNN(nn.Module):
    # input size: 224 224
    def __init__(self, num_class, input_size = 224):
        super().__init__()
        self.num_class = num_class
        self.input_size = input_size
        cnn_out_channels = 32
        self.features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1), # bs * 224 * 224 * 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # bs * 112 * 112 * 16
            nn.Conv2d(in_channels=16, out_channels=cnn_out_channels, kernel_size=3, stride=1, padding=1), # bs * 112 * 112 * 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # bs * 56 * 56 * 32
        )
        self.feature_output_dim = (input_size // 4) * (input_size // 4) * cnn_out_channels
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_output_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.num_class)
        )

    def forward(self, x):
        x = self.features_extractor(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    x = t.randn(32, 3, 224, 224)
    my_cnn = MyCNN(num_class=10, input_size=224)
    x = my_cnn(x)
    logger.info(x)
    logger.info(x.shape)