# Created by Deniz Karakay at 20.04.2023
# Filename: question_5_part2.py


import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import time
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import pickle
import models as my_models

# Hyper-parameters
EPOCH_SIZE = 30
BATCH_SIZE = 50
TRAIN_COUNT = 1

# I tested and saw that CPU is faster than GPU on M1 Pro for MLPs

# CPU
# device = torch.device("cpu")

# MPS for GPU support on M1
device = torch.device("mps")

print(f"Using device: {device}...")

init_time = time.time()

# Transformations
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    torchvision.transforms.Grayscale()
])

# Load data
train_data = torchvision.datasets.CIFAR10('../Q3/data/', train=True, download=True, transform=transform)

# Print image size and label of first image in dataset (should be 32x32 and 6)
img, label = train_data[0]
print("Sample image size: ", img.size())

# Split data into training and validation
train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)

# Load test data
test_data = torchvision.datasets.CIFAR10('../Q3/data', train=False, download=True, transform=transform)

print(f"Training data size: {len(train_data)}")
print(f"Validation data size: {len(valid_data)}")
print(f"Test data size: {len(test_data)}")

valid_generator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)

my_models_list = ['cnn4_1']

train_acc_history_total = []
train_loss_history_total = []
valid_acc_history_total = []

model_name = my_models_list[0]
print(f"Training {model_name}...")

model_init_time = time.time()
best_performance = 0
best_weights = None

for tc in range(TRAIN_COUNT):

    # Initialize model for learning rate = 0.1
    if model_name == 'cnn4_1':
        lr = 0.1
        model = my_models.CNN4().to(device)

    # Initialize model for learning rate = 0.01
    elif model_name == 'cnn4_01':
        lr = 0.01
        model = my_models.CNN4().to(device)

    # Initialize model for learning rate = 0.001
    elif model_name == 'cnn4_001':
        lr = 0.001
        model = my_models.CNN4().to(device)

    # Initialize model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    train_acc_history = []
    train_loss_history = []
    valid_acc_history = []
    test_acc_history = []

    train_generator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH_SIZE):
        # Train the model
        start_time = time.time()
        total_step = len(train_generator)

        train_loss = 0
        train_acc = 0

        if epoch == 10:
            lr = 0.01
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
        if epoch == 20:
            lr = 0.001
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)
        for i, data in enumerate(train_generator):
            model.train()
            inputs, labels = data
            train_inputs, train_labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            train_outputs = model(train_inputs)
            loss = criterion(train_outputs, train_labels)

            loss.backward()
            optimizer.step()

            model.eval()

            # Compute the loss
            train_loss += loss.item()

            # Compute the accuracy
            pred = train_outputs.argmax(dim=1, keepdim=True)
            train_total = train_labels.size(0)
            train_correct = pred.eq(train_labels.view_as(pred)).sum().item()
            train_acc += (train_correct / train_total) * 100

            if i % 10 == 0:

                # Take average of train loss and accuracy
                train_acc_history.append(train_acc / (i + 1))
                train_loss_history.append(train_loss / (i + 1))

                valid_correct = 0
                valid_total = 0
                with torch.no_grad():
                    for data in valid_generator:
                        inputs, labels = data
                        valid_inputs, valid_labels = inputs.to(device), labels.to(device)

                        # Compute the outputs and predictions
                        valid_outputs = model(valid_inputs)
                        _, valid_predicted = torch.max(valid_outputs.data, 1)

                        # Track the statistics
                        valid_total += valid_labels.size(0)
                        valid_correct += (valid_predicted == valid_labels).sum().item()

                    valid_acc = (valid_correct / valid_total) * 100
                    valid_acc_history.append(valid_acc)

        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{EPOCH_SIZE}], Epoch Time: {epoch_time:.4f} s, Train Loss: {train_loss_history[-1]:.4f}, "
            f"Train Accuracy: {train_acc_history[-1]:.3f}, Validation Accuracy: {valid_acc_history[-1]:.3f}")

    train_acc_history_total.append(train_acc_history)
    train_loss_history_total.append(train_loss_history)
    valid_acc_history_total.append(valid_acc_history)
    took_time = time.time() - model_init_time
    print(f"Training [{tc + 1}/{TRAIN_COUNT}], {took_time:.4f} s, Test Accuracy: {best_performance:.4f}")

if model_name.endswith('001'):
    # Save the results
    result_dict = {
        'name': model_name.replace('s', ''),
        'loss_curve_1': train_loss_history_total[0],
        'loss_curve_01': train_loss_history_total[1],
        'loss_curve_001': train_loss_history_total[2],
        'val_acc_curve_1': valid_acc_history_total[0],
        'val_acc_curve_01': valid_acc_history_total[1],
        'val_acc_curve_001': valid_acc_history_total[2],
    }

    # Save the results to a file
    filename = 'results/question_5_' + model_name + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(result_dict, f)

    took_time = time.time() - model_init_time
    print(f"All process for {model_name}: {took_time:.4f} s")

took_time = time.time() - init_time
print(f"All process: {took_time:.4f} s")
