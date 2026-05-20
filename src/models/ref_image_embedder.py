import torch
import torch.nn as nn


class RefImageEmbedding(nn.Module):
    def __init__(self):
        super(RefImageEmbedding, self).__init__()
        
        self.conv1 = nn.Conv2d(4, 64, kernel_size=4, padding=2, stride=1)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=4, padding=2, stride=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=4, padding=2, stride=1)
        self.conv4 = nn.Conv2d(512, 768, kernel_size=5, padding=2, stride=1)
        
        self.pool = nn.MaxPool2d(2, 1)
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = nn.functional.relu(self.conv4(x))
        
        return x
