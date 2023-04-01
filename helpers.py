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


#################################### Utility functions ####################################
def count_distinct_labels(dataset):
    labels = dataset.targets
    unique_labels = torch.unique(torch.as_tensor(labels))
    return len(unique_labels)


def get_mean_std(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(trainloader))[0]
    mean = data.mean()
    std = data.std()
    return mean, std


def get_shuffled_indices(dataset_length, train_split):
    train_data_indices = torch.randperm(dataset_length)
    train_length = int(train_split * dataset_length)
    subsample_train_indices = train_data_indices[:train_length]
    return subsample_train_indices


def get_batch_size(batch_size, train_length):
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif batch_size < 1 or batch_size > train_length:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    return batch_size


#################################### Train loaders types ####################################
def create_default_trainloaders(
    dataset, subsample_train_indices, batch_size, model_accuracy
):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(subsample_train_indices),
    )
    if model_accuracy:
        train_loader_full = train_loader
        return (train_loader, train_loader_full)
    else:
        return train_loader


def create_split_dataset_trainloaders(
    nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
):
    worker_indices = subsample_train_indices.chunk(nb_workers)
    split_dataset_dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(worker_idx),
        )
        for worker_idx in worker_indices
    ]
    if model_accuracy:
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(subsample_train_indices),
        )
        return (split_dataset_dataloaders, train_loader_full)
    else:
        return split_dataset_dataloaders


def create_split_labels_trainloaders(
    nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
):
    nb_labels = count_distinct_labels(dataset)

    label_indices = {i: [] for i in range(nb_labels)}

    for idx in subsample_train_indices:
        label = dataset[idx][1]
        label_indices[label].append(idx)

    worker_indices = [[] for _ in range(nb_workers)]
    available_labels = list(range(nb_labels))
    np.random.shuffle(available_labels)

    for i, label in enumerate(available_labels):
        worker = i % nb_workers
        worker_indices[worker].extend(label_indices[label])

    min_subset_length = len(subsample_train_indices)
    for worker in worker_indices:
        if min_subset_length > len(worker):
            min_subset_length = len(worker)

    for i, worker in enumerate(worker_indices):
        worker_indices[i] = worker[:min_subset_length]

    labels_train_loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(worker_idx),
        )
        for worker_idx in worker_indices
    ]

    if model_accuracy:
        full_labels_list = []
        for sublist in worker_indices:
            full_labels_list.extend(sublist)
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(full_labels_list),
        )
        return (labels_train_loaders, train_loader_full)
    else:
        return labels_train_loaders


def get_trainloader(
    nb_workers,
    dataset,
    subsample_train_indices,
    batch_size,
    split_dataset,
    split_labels,
    model_accuracy,
):
    if not split_dataset and not split_labels:
        train_loaders = create_default_trainloaders(
            dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif split_dataset and not split_labels:
        train_loaders = create_split_dataset_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    else:
        train_loaders = create_split_labels_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders


#################################### Dataset train loaders ####################################
def mnist_trainloaders(
    nb_workers, split_dataset, split_labels, train_split, batch_size, model_accuracy
):
    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(mnist_train)

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

    subsample_train_indices = get_shuffled_indices(len(mnist_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created MNIST trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        mnist_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def fashion_mnist_trainloaders(
    nb_workers, split_dataset, split_labels, train_split, batch_size, model_accuracy
):
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(fashion_mnist_train)

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

    subsample_train_indices = get_shuffled_indices(
        len(fashion_mnist_train), train_split
    )
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created FashionMNIST trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        fashion_mnist_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def cifar10_trainloaders(
    nb_workers, split_dataset, split_labels, train_split, batch_size, model_accuracy
):
    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar10_train)

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

    subsample_train_indices = get_shuffled_indices(len(cifar10_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created CIFAR10 trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        cifar10_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def cifar100_trainloaders(
    nb_workers, split_dataset, split_labels, train_split, batch_size, model_accuracy
):
    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar100_train)

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

    subsample_train_indices = get_shuffled_indices(len(cifar100_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created CIFAR100 trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        cifar100_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        model_accuracy,
    )
    return (train_loaders, batch_size)


#################################### TRAIN LOADER main function ####################################
def create_worker_trainloaders(
    nb_workers,
    dataset_name,
    split_dataset,
    split_labels,
    train_split,
    batch_size,
    model_accuracy,
):
    if dataset_name == "mnist":
        return mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "fashion_mnist":
        return fashion_mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "cifar10":
        return cifar10_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "cifar100":
        return cifar100_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            train_split,
            batch_size,
            model_accuracy,
        )
    else:
        print("Error Unkown dataset")
        exit()


#################################### Dataset test loaders ####################################
def mnist_testloader(batch_size):
    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(mnist_train)

    mnist_test = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False
    )
    return test_loader, count_distinct_labels(mnist_test)


def fashion_mnist_testloader(batch_size):
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(fashion_mnist_train)

    fashion_mnist_test = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        fashion_mnist_test, batch_size=batch_size, shuffle=False
    )
    return test_loader, count_distinct_labels(fashion_mnist_test)


def cifar10_testloader(batch_size):
    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar10_train)

    cifar10_test = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=batch_size, shuffle=False
    )
    return test_loader, count_distinct_labels(cifar10_test)


def cifar100_testloader(batch_size):
    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar100_train)

    cifar100_test = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=False,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        cifar100_test, batch_size=batch_size, shuffle=False
    )
    return test_loader, count_distinct_labels(cifar100_test)


#################################### TEST LOADER main function ####################################
def create_testloader(model_path, batch_size):
    if "fashion_mnist" in model_path:
        print("Created FashionMNIST testloader \n")
        return fashion_mnist_testloader(batch_size)
    elif "mnist" in model_path:
        print("Created MNIST testloader \n")
        return mnist_testloader(batch_size)
    elif "cifar100" in model_path:
        print("Created CIFAR100 testloader \n")
        return cifar100_testloader(batch_size)
    elif "cifar10" in model_path:
        print("Created CIFAR10 testloader \n")
        return cifar10_testloader(batch_size)
    else:
        print("Error Unkown dataset")
        exit()
