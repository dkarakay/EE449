# Created by Deniz Karakay at 18.04.2023
# Filename: question_4.py

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import time
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from Q4 import models as my_models
from utils.utils import visualizeWeights

# Hyper-parameters
EPOCH_SIZE = 15
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

# Sigmoid and ReLU based models
my_models_list = ['mlp1', 'mlp1s', 'mlp2', 'mlp2s', 'cnn3', 'cnn3s', 'cnn4', 'cnn4s', 'cnn5', 'cnn5s']

relu_loss_history = []
relu_grad_history = []
sigmoid_loss_history = []
sigmoid_grad_history = []

for model_name in my_models_list:
    if model_name.startswith('mlp'):
        continue

    print(f"Training {model_name}...")

    model_init_time = time.time()
    best_performance = 0
    best_weights = None

    for tc in range(TRAIN_COUNT):

        if model_name == 'mlp1':
            model = my_models.MLP1(1024, 32, 10).to(device)
        elif model_name == 'mlp1s':
            model = my_models.MLP1S(1024, 32, 10).to(device)
        elif model_name == 'mlp2':
            model = my_models.MLP2(1024, 32, 64, 10).to(device)
        elif model_name == 'mlp2s':
            model = my_models.MLP2S(1024, 32, 64, 10).to(device)
        elif model_name == 'cnn3':
            model = my_models.CNN3().to(device)
        elif model_name == 'cnn3s':
            model = my_models.CNN3S().to(device)
        elif model_name == 'cnn4':
            model = my_models.CNN4().to(device)
        elif model_name == 'cnn4s':
            model = my_models.CNN4S().to(device)
        elif model_name == 'cnn5':
            model = my_models.CNN5().to(device)
        elif model_name == 'cnn5s':
            model = my_models.CNN5S().to(device)

        # Initialize model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0)

        train_generator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCH_SIZE):
            # Train the model
            start_time = time.time()
            total_step = len(train_generator)

            train_loss = 0
            train_acc = 0

            for i, data in enumerate(train_generator):
                model.train()
                inputs, labels = data
                train_inputs, train_labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                train_outputs = model(train_inputs)

                # Define loss
                loss = criterion(train_outputs, train_labels)

                # Backward
                loss.backward()

                # Update the parameters
                optimizer.step()

                model.eval()

                # Compute the loss
                train_loss += loss.item()

                if i % 10 == 0:

                    # Move model to CPU to compute the gradient
                    model.to('cpu')
                    # Get the gradient of the first layer
                    weight = model.first.weight.grad
                    # To indicate the gradient of the first layer better on the plot
                    train_grad = np.linalg.norm(weight)
                    # Move model back to GPU
                    model.to(device)

                    # Sigmoid based models are saved in a different list
                    if model_name.endswith('s'):
                        sigmoid_loss_history.append(train_loss / (i + 1))
                        sigmoid_grad_history.append(train_grad)

                    # ReLU based models are saved in a different list
                    else:
                        relu_loss_history.append(train_loss / (i + 1))
                        relu_grad_history.append(train_grad)

            epoch_time = time.time() - start_time
            print(
                f"Epoch [{epoch + 1}/{EPOCH_SIZE}], Epoch Time: {epoch_time:.4f} s  Gradient: {train_grad:.4f}  Loss: {train_loss / total_step:.4f}")

        took_time = time.time() - model_init_time
        print(f"Training [{tc + 1}/{TRAIN_COUNT}], {took_time:.4f} s, Test Accuracy: {best_performance:.4f}")

    if model_name.endswith('s'):
        # Save the results
        result_dict = {
            'name': model_name.replace('s', ''),
            'relu_loss_curve': relu_loss_history,
            'relu_grad_curve': relu_grad_history,
            'sigmoid_loss_curve': sigmoid_loss_history,
            'sigmoid_grad_curve': sigmoid_grad_history,
        }

        relu_loss_history = []
        relu_grad_history = []
        sigmoid_loss_history = []
        sigmoid_grad_history = []

        # Save the results to a file
        filename = 'results/question_4_' + model_name.replace('s', '') + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(result_dict, f)

        took_time = time.time() - model_init_time
        print(f"All process for {model_name}: {took_time:.4f} s")

took_time = time.time() - init_time
print(f"All process: {took_time:.4f} s")
