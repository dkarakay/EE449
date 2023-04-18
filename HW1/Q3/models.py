# Created by Deniz Karakay at 17.04.2023
# Filename: models.py
import torch.nn as nn


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP1, self).__init__()
        self.input_size = input_size
        self.first = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.first(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output


class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.first = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.pred_layer = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.first(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.pred_layer(x)
        return output


class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()

        self.first = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding='valid')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding='valid')
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.pred_layer = nn.Linear(in_features=144, out_features=10)

    def forward(self, x):
        x = self.first(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        output = self.pred_layer(x)
        return output


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


class CNN5(nn.Module):
    def __init__(self):
        super(CNN5, self).__init__()

        self.first = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding='valid')
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding='valid')
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.pred_layer = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.first(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        x = x.view(x.size(0), -1)
        output = self.pred_layer(x)
        return output
