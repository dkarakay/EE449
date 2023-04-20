# Created by Deniz Karakay at 17.04.2023
# Filename: models.py
import torch.nn as nn


class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()

        self.first = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding='valid')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.pred_layer = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.first(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        output = self.pred_layer(x)
        return output
