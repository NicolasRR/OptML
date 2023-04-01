import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

DEFAULT_BATCH_SIZE = 32  # 1 == SGD, >1 MINI BATCH SGD


#################################### NET ####################################
class CNN_MNIST(nn.Module):  # for MNIST and Fashion MNIST
    def __init__(self):
        super(CNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


class CNN_CIFAR(nn.Module):  # for CIFAR10 and CIFAR100
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output


#################################### TRAIN LOADERS ####################################
def create_worker_trainloaders(
    nb_workers,
    dataset_name,
    split_dataset,
    digits_mode,
    train_split,
    batch_size,
    model_accuracy,
):
    if dataset_name == "mnist":
        mnist_train = torchvision.datasets.MNIST(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

        trainloader = torch.utils.data.DataLoader(
            mnist_train, batch_size=len(mnist_train)
        )
        data = next(iter(trainloader))[0]
        mean = data.mean()
        std = data.std()

        mnist_train = torchvision.datasets.MNIST(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )

        # Shuffle and split the train_data
        train_data_indices = torch.randperm(len(mnist_train))
        train_length = int(train_split * len(mnist_train))
        subsample_train_indices = train_data_indices[:train_length]

        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
            print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
        elif batch_size < 1 or batch_size > train_length:
            print("Forbidden value !!! batch_size must be between [1,len(train set)]")
            exit()

        if not split_dataset and not digits_mode:  # workers sharing samples
            train_loader = DataLoader(
                mnist_train,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(subsample_train_indices),
            )
            if model_accuracy:
                train_loader_full = train_loader
                return (train_loader, train_loader_full)
            else:
                return train_loader

        elif split_dataset and not digits_mode:
            worker_indices = subsample_train_indices.chunk(
                nb_workers
            )  # Split the train_indices based on the number of workers (world_size - 1)
            split_dataset_dataloaders = [
                DataLoader(
                    mnist_train,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(worker_idx),
                )
                for worker_idx in worker_indices
            ]
            if model_accuracy:
                train_loader_full = DataLoader(
                    mnist_train,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(subsample_train_indices),
                )
                return (split_dataset_dataloaders, train_loader_full)
            else:
                return split_dataset_dataloaders

        elif digits_mode:
            digit_indices = {i: [] for i in range(10)}
            for idx in subsample_train_indices:
                digit = mnist_train[idx][1]
                digit_indices[digit].append(idx)
            worker_indices = [[] for _ in range(nb_workers)]
            available_digits = list(range(10))
            np.random.shuffle(available_digits)

            for i, digit in enumerate(available_digits):
                worker = i % nb_workers
                worker_indices[worker].extend(digit_indices[digit])

            min_subset_length = len(subsample_train_indices)
            for worker in worker_indices:
                if min_subset_length > len(worker):
                    min_subset_length = len(worker)

            for i, worker in enumerate(worker_indices):
                worker_indices[i] = worker[:min_subset_length]

            digit_train_loaders = [
                DataLoader(
                    mnist_train,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(worker_idx),
                )
                for worker_idx in worker_indices
            ]

            if model_accuracy:
                full_digits_list = []
                for sublist in worker_indices:
                    full_digits_list.extend(sublist)
                train_loader_full = DataLoader(
                    mnist_train,
                    batch_size=batch_size,
                    sampler=SubsetRandomSampler(full_digits_list),
                )
                return (digit_train_loaders, train_loader_full)
            else:
                return digit_train_loaders

        elif dataset_name == "fashion_mnist":
            fashion_mnist_train = torchvision.datasets.FashionMNIST(
                root="./data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            )

            trainloader = torch.utils.data.DataLoader(
                fashion_mnist_train, batch_size=len(fashion_mnist_train)
            )
            data = next(iter(trainloader))[0]
            mean = data.mean()
            std = data.std()

            fashion_mnist_train = torchvision.datasets.FashionMNIST(
                "data/",
                download=True,
                train=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                    ]
                ),
            )

            train_data_indices = torch.randperm(len(fashion_mnist_train))
            train_length = int(train_split * len(fashion_mnist_train))
            subsample_train_indices = train_data_indices[:train_length]

            if batch_size is None:
                batch_size = DEFAULT_BATCH_SIZE
                print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
            elif batch_size < 1 or batch_size > train_length:
                print(
                    "Forbidden value !!! batch_size must be between [1,len(train set)]"
                )
                exit()

            print("fashion mnist trainloaders not implemented yet")
            exit()

        elif dataset_name == "cifar10":
            cifar10_train = torchvision.datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            )

            trainloader = torch.utils.data.DataLoader(
                cifar10_train, batch_size=len(cifar10_train)
            )
            data = next(iter(trainloader))[0]
            mean = data.mean()
            std = data.std()

            cifar10_train = torchvision.datasets.CIFAR10(
                "data/",
                download=True,
                train=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                    ]
                ),
            )

            train_data_indices = torch.randperm(len(cifar10_train))
            train_length = int(train_split * len(cifar10_train))
            subsample_train_indices = train_data_indices[:train_length]

            if batch_size is None:
                batch_size = DEFAULT_BATCH_SIZE
                print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
            elif batch_size < 1 or batch_size > train_length:
                print(
                    "Forbidden value !!! batch_size must be between [1,len(train set)]"
                )
                exit()

            print("cifar10 trainloaders not implemented yet")
            exit()

        elif dataset_name == "cifar100":
            cifar100_train = torchvision.datasets.CIFAR100(
                root="./data",
                train=True,
                download=True,
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()]
                ),
            )

            trainloader = torch.utils.data.DataLoader(
                cifar100_train, batch_size=len(cifar100_train)
            )
            data = next(iter(trainloader))[0]
            mean = data.mean()
            std = data.std()

            cifar100_train = torchvision.datasets.CIFAR100(
                "data/",
                download=True,
                train=True,
                transform=torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                    ]
                ),
            )

            print("cifar100 mnist trainloaders not implemented yet")
            exit()

        else:
            print("Error Unkown dataset")
            exit()
