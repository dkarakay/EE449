# Created by Deniz Karakay at 17.04.2023
# Filename: models.py
import torch.nn as nn


class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP1, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output


class MLP2(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_classes):
        super(MLP2, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2, bias=False)
        self.pred_layer = nn.Linear(hidden_size_2, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        output = self.pred_layer(x)
        return output
