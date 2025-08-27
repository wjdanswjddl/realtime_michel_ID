# networks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

import numpy as np
import random
import json
import math
import datetime


class Model3Class(nn.Module):
    def __init__(self):
        super(Model3Class, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 584, 256) -> (16, 584, 256)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 292, 128)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32, 292, 128)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 146, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> (64, 146, 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 73, 32)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 73 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # Output: 3 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        logits = self.fc_layers(x)
        return logits
