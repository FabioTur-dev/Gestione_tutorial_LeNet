import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input: 1 x 32 x 32 (MNIST 28x28 padded to 32x32)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # -> (6, 28, 28)
        x = F.max_pool2d(x, 2)         # -> (6, 14, 14)
        x = F.relu(self.conv2(x))      # -> (16, 10, 10)
        x = F.max_pool2d(x, 2)         # -> (16, 5, 5)
        x = x.view(x.size(0), -1)      # -> (batch, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
