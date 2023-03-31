import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def split_indices_by_digits(indices, num_workers):
    digit_indices = {i: [] for i in range(10)}
    
    for idx in indices:
        digit = train_data[idx][1]
        digit_indices[digit].append(idx)
    
    worker_indices = [[] for _ in range(num_workers)]
    available_digits = list(range(10))
    np.random.shuffle(available_digits)

    for i, digit in enumerate(available_digits):
        print(i, i % num_workers, digit)
        worker = i % num_workers
        worker_indices[worker].extend(digit_indices[digit])

    print(len(worker_indices[0]), len(worker_indices[1]), len(worker_indices[2]), len(worker_indices[3]), len(worker_indices[4]))
    min_subset_length = len(indices)
    for worker in worker_indices:
        if min_subset_length > len(worker):
            min_subset_length = len(worker)
    print(min_subset_length)

    for i, worker in enumerate(worker_indices):
        worker_indices[i] = worker[:min_subset_length]

    print(len(worker_indices[0]), len(worker_indices[1]), len(worker_indices[2]), len(worker_indices[3]), len(worker_indices[4]))

    return worker_indices

def verify_dataloader_digits(data_loader, worker_idx):
    unique_digits = set()
    for _, labels in data_loader:
        unique_digits.update(labels.numpy())

    expected_digits = set()
    for idx in worker_indices[worker_idx]:
        expected_digits.add(train_data[idx][1])

    unique_digits = sorted(list(unique_digits))
    expected_digits = sorted(list(expected_digits))
    return unique_digits == expected_digits


# MNIST dataset
train_data = torchvision.datasets.MNIST('data/',
                                        download=True,
                                        train=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))

train_split = 1.0  # Adjust this value if you want to use a subset of the training data
train_data_indices = torch.randperm(len(train_data))
train_length = int(train_split * len(train_data))
subsample_train_indices = train_data_indices[:train_length]

num_workers = 5  # Change this value to 2, 5, or 10 depending on your requirements
worker_indices = split_indices_by_digits(subsample_train_indices, num_workers)

batch_size = 64  # Set the desired batch size
data_loaders = [DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(worker_idx)) for worker_idx in worker_indices]

# Verify the DataLoader digits
for i, data_loader in enumerate(data_loaders):
    result = verify_dataloader_digits(data_loader, i)
    print(f"DataLoader {i+1}: Verification result: {result}")