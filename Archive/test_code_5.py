import torch
from torchvision import datasets


def count_distinct_labels(trainset):
    labels = (
        trainset.targets
    )  # For PyTorch dataset, trainset.targets contains the labels
    unique_labels = torch.unique(
        torch.as_tensor(labels)
    )  # Convert the list to a tensor and find the unique labels
    return len(unique_labels)  # Return the number of unique labels


# Load the datasets using PyTorch
mnist_train = datasets.MNIST(root="./data", train=True, download=True)
fashion_mnist_train = datasets.FashionMNIST(root="./data", train=True, download=True)
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True)
cifar100_train = datasets.CIFAR100(root="./data", train=True, download=True)

# Count the distinct labels for each dataset
mnist_labels = count_distinct_labels(mnist_train)
fashion_mnist_labels = count_distinct_labels(fashion_mnist_train)
cifar10_labels = count_distinct_labels(cifar10_train)
cifar100_labels = count_distinct_labels(cifar100_train)

# Print the results
print("MNIST distinct labels:", mnist_labels)
print("Fashion MNIST distinct labels:", fashion_mnist_labels)
print("CIFAR-10 distinct labels:", cifar10_labels)
print("CIFAR-100 distinct labels:", cifar100_labels)
