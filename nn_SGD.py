# %% PACKAGES
import os
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F


# %% NETWORK CNN FOR CLASSIFICATION
class Initial_Net(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.conv1 = nn.Conv2d(num_input, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_output)
        self.max_pool = nn.AdaptiveMaxPool2d((12, 12))

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# %% TRANING PARAMETERS
def train_model_SGD(
    network, learning_rate, momentum, epochs, train_loader, print_every
):
    # Set the network and the loss function
    model = network
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    # Set the number of epochs to train for
    n_epochs = epochs
    start_time = time.time()
    for epoch in range(n_epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0
        last_rn_loss = 0
        correct_predictions = 0
        last_correct_pred = 0
        # Loop over the training data
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move the data and target to the device
            data, target = data, target
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Compute the loss
            loss = criterion(output, target)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Update the running loss and correct predictions
            running_loss += loss.item()
            _, predictions = torch.max(output.data, 1)
            correct_predictions += (predictions == target).sum().item()
            if batch_idx % print_every == 0:
                # print running loss and correct prediction over the last n batches
                print(
                    "Epochs : {}  , Batch : {},  Loss : {},  Accruacy : {}".format(
                        epoch + 1,
                        batch_idx,
                        running_loss - last_rn_loss,
                        (correct_predictions - last_correct_pred) / print_every,
                    )
                )
                last_correct_pred = correct_predictions
                last_rn_loss = running_loss

        # Compute the average loss and accuracy for this epoch
        avg_loss = running_loss / len(train_loader.dataset)
        accuracy = correct_predictions / len(train_loader.dataset)

        # Print the loss for this epoch
        print(
            "Epoch: {} Loss: {:.6f} Accuracy: {:.2f}%".format(
                epoch + 1, avg_loss, accuracy * 100
            )
        )
    training_time = time.time() - start_time
    print(f"Training time: {training_time} seconds")

    return model


if __name__ == "__main__":
    # Ask for the dataset to use
    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transforms.ToTensor(), download=True
    )

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Ask if testing resutls are required
    data_sample, _ = train_dataset[0]
    input_dim = data_sample.shape.numel()
    output_dim = len(train_dataset.classes)

    network = Initial_Net(num_input=3, num_output=output_dim)
    # Parameters
    lr = 0.0001
    momentum = 0.0001
    epochs = 10
    print_every = 10
    trained_model = train_model_SGD(
        network, lr, momentum, epochs, train_loader, print_every
    )


# %%
