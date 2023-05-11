import argparse
from sklearn.model_selection import KFold
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from common import _get_model, create_worker_trainloaders, LOSS_FUNC

DEFAULT_K_SPLITS = 5

parser = argparse.ArgumentParser(description="KFold Cross Validation ")
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
    help="""Dataset name for KFold CV: mnist, FashionMNIST, CIFAR10, CIFAR100""",
)
parser.add_argument(
    "--k_splits",
    type=int,
    default=DEFAULT_K_SPLITS,
    help="""The number of splits for KFold CV.""",
)
args = parser.parse_args()

model = _get_model(args.dataset, LOSS_FUNC)

if args.k_splits < 2:
    print("Forbiden value!!! --k_split should be >= 2")
    exit()

loader = create_worker_trainloaders(
    args.dataset, train_split=1, batch_size=100, model_accuracy=False
)[0]
indices = []
for batch_indices, _ in loader:
    indices.extend(batch_indices.numpy())
indices = np.array(indices)

kf = KFold(n_splits=args.k_splits)

learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
momentums = [0.9, 0.95, 0.99]
batch_sizes = [32, 64, 100, 128]
epochs = [2, 4, 6, 8, 10]

total_steps = len(epochs) * len(learning_rates) * len(momentums) * len(batch_sizes)
current_step = 0
avg_losses = np.zeros(
    (len(epochs), len(learning_rates), len(momentums), len(batch_sizes))
)

for epoch_index, epoch in enumerate(epochs):
    for lr_index, learning_rate in enumerate(learning_rates):
        for momentum_index, momentum in enumerate(momentums):
            for batch_size_index, batch_size in enumerate(batch_sizes):
                avg_loss = 0.0
                for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
                    print(
                        f"Step: {current_step+1}/{total_steps}, Fold: {fold + 1}/{kf.get_n_splits()}"
                    )
                    model = _get_model("mnist", LOSS_FUNC)
                    optimizer = optim.SGD(
                        model.parameters(), lr=learning_rate, momentum=momentum
                    )
                    train_sampler = SubsetRandomSampler(train_indices)
                    val_sampler = SubsetRandomSampler(val_indices)
                    train_dataloader = DataLoader(
                        loader.dataset, batch_size=batch_size, sampler=train_sampler
                    )
                    val_dataloader = DataLoader(
                        loader.dataset, batch_size=batch_size, sampler=val_sampler
                    )

                    for _ in range(epoch):
                        for data, target in train_dataloader:
                            optimizer.zero_grad()
                            output = model(data)
                            loss = LOSS_FUNC(output, target)
                            loss.backward()
                            optimizer.step()

                    val_loss = 0.0
                    val_count = 0
                    with torch.no_grad():
                        for data, target in val_dataloader:
                            output = model(data)
                            loss = LOSS_FUNC(output, target)
                            val_loss += loss.item()
                            val_count += 1
                    avg_loss += val_loss / val_count

                avg_loss /= kf.get_n_splits()
                avg_losses[
                    epoch_index, lr_index, momentum_index, batch_size_index
                ] = avg_loss
                current_step += 1

min_loss_index = np.unravel_index(np.argmin(avg_losses, axis=None), avg_losses.shape)
min_loss_value = avg_losses[min_loss_index]
print(
    f"\nBest parameters: epochs={epochs[min_loss_index[0]]}, learning_rate={learning_rates[min_loss_index[1]]}, momentum={momentums[min_loss_index[2]]}, batch_size={batch_sizes[min_loss_index[3]]}, loss={min_loss_value}"
)

index = pd.MultiIndex.from_product(
    [epochs, learning_rates, momentums, batch_sizes],
    names=["epochs", "learning_rate", "momentum", "batch_size"],
)
df = pd.DataFrame(avg_losses.flatten(), index=index, columns=["Average Loss"])
print(df.reset_index().to_string(index=False))
