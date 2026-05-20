import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthPredicter(nn.Module):
    r"""
    Predict the hand mask of unet output.
    """

    def __init__(self):
        super(DepthPredicter, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels=320, out_channels=128, kernel_size=2, stride=2)
        self.silu1 = nn.SiLU()
        self.conv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.silu2 = nn.SiLU()
        self.conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.silu1(self.conv1(x))
        x = self.silu2(self.conv2(x))
        x = self.conv3(x)

        return x
