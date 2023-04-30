import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import logging
import queue
from multiprocessing import Manager


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


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
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


"""class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x"""


class CNN_CIFAR100(nn.Module):
    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)

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


"""class CNN_CIFAR100(nn.Module):
    def __init__(self):
        super(CNN_CIFAR100, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""


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


def create_split_labels_unscaled_trainloaders(
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
    split_labels_unscaled,
    model_accuracy,
):
    if not split_dataset and not split_labels and not split_labels_unscaled:
        train_loaders = create_default_trainloaders(
            dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif split_dataset and not split_labels and not split_labels_unscaled:
        train_loaders = create_split_dataset_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif split_labels and not split_labels_unscaled:
        train_loaders = create_split_labels_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders
    else:
        train_loaders = create_split_labels_unscaled_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

############################## Loggers

def setup_logger(log_queue):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    qh = QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    qh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(qh)

    return logger

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

def log_writer(log_queue, output_folder = "."):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    with open(os.path.join(output_folder,'log.log'), 'w') as log_file:
        while True:
            try:
                record = log_queue.get(timeout=1) 
                if record is None:
                    break
                msg = formatter.format(record)
                log_file.write(msg + "\n")
            except queue.Empty:
                continue

#################################### Dataset train loaders ####################################
def mnist_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
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
        split_labels_unscaled,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def fashion_mnist_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
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
        split_labels_unscaled,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def cifar10_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
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
        split_labels_unscaled,
        model_accuracy,
    )
    return (train_loaders, batch_size)


def cifar100_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
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
        split_labels_unscaled,
        model_accuracy,
    )
    return (train_loaders, batch_size)


#################################### TRAIN LOADER main function ####################################
def create_worker_trainloaders(
    nb_workers,
    dataset_name,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
):
    if dataset_name == "mnist":
        return mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "fashion_mnist":
        return fashion_mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "cifar10":
        return cifar10_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
        )
    elif dataset_name == "cifar100":
        return cifar100_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
        )
    else:
        print("Error Unkown dataset")
        exit()


#################################### Dataset test loaders ####################################
def mnist_testloader(batch_size, test=True):
    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(mnist_train)

    if test:
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
    else:
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
        test_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def fashion_mnist_testloader(batch_size, test=True):
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(fashion_mnist_train)

    if test:
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
    else:
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
        test_loader = torch.utils.data.DataLoader(
            fashion_mnist_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def cifar10_testloader(batch_size, test=True):
    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar10_train)
    if test:
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
    else:
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
        test_loader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def cifar100_testloader(batch_size, test=True):
    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar100_train)
    if test:
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
    else:
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
        test_loader = torch.utils.data.DataLoader(
            cifar100_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


#################################### TEST LOADER main function ####################################
def create_testloader(model_path, batch_size):
    if "fashion_mnist" in model_path:
        # print("Created FashionMNIST testloader \n")
        return fashion_mnist_testloader(batch_size)
    elif "mnist" in model_path:
        # print("Created MNIST testloader \n")
        return mnist_testloader(batch_size)
    elif "cifar100" in model_path:
        # print("Created CIFAR100 testloader \n")
        return cifar100_testloader(batch_size)
    elif "cifar10" in model_path:
        # print("Created CIFAR10 testloader \n")
        return cifar10_testloader(batch_size)
    else:
        print("Error Unkown dataset")
        exit()


def create_trainloader(model_path, batch_size):
    if "fashion_mnist" in model_path:
        # print("Created FashionMNIST trainloader \n")
        return fashion_mnist_testloader(batch_size, test=False)
    elif "mnist" in model_path:
        # print("Created MNIST trainloader \n")
        return mnist_testloader(batch_size, test=False)
    elif "cifar100" in model_path:
        # print("Created CIFAR100 trainloader \n")
        return cifar100_testloader(batch_size, test=False)
    elif "cifar10" in model_path:
        # print("Created CIFAR10 trainloader \n")
        return cifar10_testloader(batch_size, test=False)
    else:
        print("Error Unkown dataset")
        exit()
