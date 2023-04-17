# Created by Deniz Karakay at 17.04.2023
# Filename: question_3.py

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import time
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split

from Q3 import models as my_models

# Hyper-parameters
EPOCH_SIZE = 15
BATCH_SIZE = 50
TRAIN_COUNT = 10

# I tested and saw that CPU is faster than GPU on M1 Pro

# MPS for GPU support on M1
# device = torch.device("mps")

# CPU
device = torch.device("cpu")
print(device)

init_time = time.time()

# Transformations
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    torchvision.transforms.Grayscale()
])

# Load data
train_data = torchvision.datasets.CIFAR10('data/', train=True, download=True, transform=transform)

# Print image size and label of first image in dataset (should be 32x32 and 6)
img, label = train_data[0]
print("Sample image size: ", img.size())

# Split data into training and validation
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Load test data
test_data = torchvision.datasets.CIFAR10('data/', train=False, transform=transform)

train_acc_history = []
train_loss_history = []
valid_acc_history = []
test_acc_history = []
best_performance = 0

mlp1 = my_models.MLP1(1024, 32, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp1.parameters())

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(valid_data)}")
print(f"Test data size: {len(test_data)}")

valid_generator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

for tc in range(TRAIN_COUNT):
    for epoch in range(EPOCH_SIZE):
        start_time = time.time()  # start timer
        train_generator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        total_step = len(train_generator)
        print(f"Epoch: {epoch}/{EPOCH_SIZE} - Step: {total_step}")

        for i, data in enumerate(train_generator):
            mlp1.train()
            inputs, labels = data
            train_inputs, train_labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            train_outputs = mlp1(train_inputs)
            loss = criterion(train_outputs, train_labels)

            loss.backward()
            optimizer.step()

            if i % 10 == 9:
                mlp1.eval()
                # Compute training accuracy
                _, train_pred = torch.max(train_outputs, 1)
                train_size = train_labels.size(0)
                train_corrects = torch.sum(train_pred == train_labels.data)

                train_acc = train_corrects / train_size

                # Save training loss
                train_loss = loss.item()

                valid_correct = 0
                valid_total = 0
                with torch.no_grad():
                    for data in valid_generator:
                        inputs, labels = data
                        valid_inputs, valid_labels = inputs.to(device), labels.to(device)

                        # Compute the outputs and predictions
                        valid_outputs = mlp1(valid_inputs)
                        _, valid_predicted = torch.max(valid_outputs.data, 1)

                        # Track the statistics
                        valid_total += valid_labels.size(0)
                        valid_correct += (valid_predicted == valid_labels).sum().item()

                valid_acc = valid_correct / valid_total

                valid_acc_history.append(valid_acc)
                train_acc_history.append(train_acc)
                train_loss_history.append(train_loss)
                # print(f'Train accuracy: {train_acc:.3f}')

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{EPOCH_SIZE}], Epoch Time: {epoch_time:.4f} s, Validation Accuracy: {valid_acc:.4f}")

        # Evaluate the model on test set
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            test_generator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
            mlp1.eval()
            for data in test_generator:
                inputs, labels = data
                test_inputs, test_labels = inputs.to(device), labels.to(device)

                # Compute the outputs and predictions
                test_outputs = mlp1(test_inputs)
                _, test_predicted = torch.max(test_outputs.data, 1)

                # Track the statistics
                test_total += test_labels.size(0)
                test_correct += (test_predicted == test_labels).sum().item()
            test_acc = test_correct / test_total
            print(f'Test accuracy: {test_acc:.3f}')
        test_acc_history.append(test_acc)

        if test_acc > best_performance:
            best_performance = test_acc
            best_weight = mlp1.fc1.weight.data.cpu().numpy()

    print(f"Best performance: {best_performance:.4f}")
    took_time = time.time() - init_time
    print(f"Process: {took_time:.4f} s")
    print(f"Best weight: {best_weight}")

took_time = time.time() - init_time
print(f"All process: {took_time:.4f} s")
