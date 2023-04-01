import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


def split_indices_evenly(indices, num_workers):
    samples_per_worker = len(indices) // num_workers
    worker_indices = [
        indices[i * samples_per_worker : (i + 1) * samples_per_worker]
        for i in range(num_workers)
    ]

    # If there are any remaining samples, distribute them among the workers
    remaining_samples = len(indices) % num_workers
    for i in range(remaining_samples):
        worker_indices[i] = torch.cat(
            (worker_indices[i], indices[-(i + 1)].unsqueeze(0))
        )
    worker_indices = subsample_train_indices.chunk(num_workers)
    print(len(worker_indices[0]), len(worker_indices[1]), len(worker_indices[2]))
    return worker_indices


# MNIST dataset
train_data = torchvision.datasets.MNIST(
    "data/",
    download=True,
    train=True,
    transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    ),
)

train_split = 1.0  # Adjust this value if you want to use a subset of the training data
train_data_indices = torch.randperm(len(train_data))
train_length = int(train_split * len(train_data))
subsample_train_indices = train_data_indices[:train_length]

num_workers = 3  # Change this value depending on the number of workers you want
worker_indices = split_indices_evenly(subsample_train_indices, num_workers)

batch_size = 64  # Set the desired batch size
data_loaders = [
    DataLoader(
        train_data, batch_size=batch_size, sampler=SubsetRandomSampler(worker_idx)
    )
    for worker_idx in worker_indices
]


def verify_no_shared_samples(worker_indices):
    all_indices = set()
    for worker_idx in worker_indices:
        current_indices = set(worker_idx.numpy())
        if any(idx in all_indices for idx in current_indices):
            return False
        all_indices.update(current_indices)
    return True


# Verify that there are no shared samples among the DataLoaders
verification_result = verify_no_shared_samples(worker_indices)
print(f"Verification result: {verification_result}")
