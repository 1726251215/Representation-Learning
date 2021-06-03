from typing import Any

import torch
import torch.nn as nn


class CNN(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, w, h):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=3)  # b 16 w//2 h//2
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)  # b 16 w//2 h//2
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)  # b 64 w//2 h//2
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2)  # b 64 w//4 h//4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=3, padding=1, stride=2)  # b 64 w//4 h//4
        self.bn5 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.out_layer = nn.Linear(w // 4 * (h // 4) * 16, 10)  # b 1 1 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = x.view(x.shape[0], -1)
        return self.softmax(self.out_layer(x).squeeze())
