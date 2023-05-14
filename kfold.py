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
DEFAULT_MOMENTUM = 0.9

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
parser.add_argument(
    "--alr",
    action="store_true",
    help="""Adam instead of vanilla SGD.""",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=None,
    nargs="?",
    const=DEFAULT_MOMENTUM,
    help="""For SGD optimizer, use this momentum for KFold CV.""",
)
parser.add_argument(
    "--light_model",
    action="store_true",
    help="""If set, will use the light CNN models.""",
)
args = parser.parse_args()

if args.k_splits < 2:
    print("Forbiden value!!! --k_split should be >= 2")
    exit()

if args.alr:
    print("Using Adam as optimizer.")
else:
    print("Using SGD as optimizer.")

if args.light_model:
    print("Using light CNN models.")

if args.momentum is not None and not args.alr:
    print(f"Using momentum: {args.momentum}")

print(f"Running Kfold for {args.dataset}, using {args.k_splits} splits.")

loader = create_worker_trainloaders(
    args.dataset, train_split=1, batch_size=100, model_accuracy=False
)[0]
indices = []
for batch_indices, _ in loader:
    indices.extend(batch_indices.numpy())
indices = np.array(indices)

kf = KFold(n_splits=args.k_splits)

if args.alr is False:
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1]
else:
    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
if args.momentum is None and args.alr is False:
    momentums = [0.9, 0.95, 0.99]
else:
    momentums = [args.momentum]
batch_sizes = [32, 64, 128]
epochs = [2, 4, 6]

if args.alr == False:
    total_steps = len(epochs) * len(batch_sizes) * len(learning_rates) * len(momentums)
    avg_losses = np.zeros(
        (
            len(epochs),
            len(batch_sizes),
            len(learning_rates),
            len(momentums),
        )
    )
else:
    total_steps = len(epochs) * len(batch_sizes) * len(learning_rates)
    avg_losses = np.zeros(
        (
            len(epochs),
            len(batch_sizes),
            len(learning_rates),
        )
    )

current_step = 0


def kfold_loop(
    kf,
    indices,
    alr,
    learning_rate,
    batch_size,
    epoch,
    dataset,
    total_steps,
    avg_losses,
    epoch_index,
    lr_index,
    batch_size_index,
    light_model,
    momentum_index=None,
    momentum=None,
):
    avg_loss = 0.0
    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        model = _get_model(args.dataset, LOSS_FUNC, light_model)
        if alr == False:
            optimizer = optim.SGD(
                model.parameters(), lr=learning_rate, momentum=momentum
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler
        )
        val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

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
        print(
            f"Fold: {fold + 1}/{kf.get_n_splits()}, fold_loss: {val_loss / val_count}"
        )

    avg_loss /= kf.get_n_splits()

    if alr == False:
        avg_losses[
            epoch_index,
            batch_size_index,
            lr_index,
            momentum_index,
        ] = avg_loss
    else:
        avg_losses[
            epoch_index,
            batch_size_index,
            lr_index,
        ] = avg_loss

    current_step += 1
    print(
        f"Step: {current_step}/{total_steps}, avg loss: {avg_loss}, epoch:{epoch}, lr:{learning_rate}, batch_size: {batch_size}"
    )
    return current_step, avg_losses


for epoch_index, epoch in enumerate(epochs):
    for batch_size_index, batch_size in enumerate(batch_sizes):
        for lr_index, learning_rate in enumerate(learning_rates):
            if args.alr == False:
                for momentum_index, momentum in enumerate(momentums):
                    current_step, avg_losses = kfold_loop(
                        kf,
                        indices,
                        args.alr,
                        learning_rate,
                        batch_size,
                        epoch,
                        loader.dataset,
                        total_steps,
                        avg_losses,
                        epoch_index,
                        lr_index,
                        batch_size_index,
                        args.light_model,
                        momentum_index=momentum_index,
                        momentum=momentum,
                    )
            else:
                current_step, avg_losses = kfold_loop(
                    kf,
                    indices,
                    args.alr,
                    learning_rate,
                    batch_size,
                    epoch,
                    loader.dataset,
                    total_steps,
                    avg_losses,
                    epoch_index,
                    lr_index,
                    batch_size_index,
                    args.light_model,
                    momentum_index=None,
                    momentum=None,
                )

min_loss_index = np.unravel_index(np.argmin(avg_losses, axis=None), avg_losses.shape)
min_loss_value = avg_losses[min_loss_index]

if args.alr == False:
    print(
        f"\nBest parameters for {args.dataset} (SGD): epochs={epochs[min_loss_index[0]]}, batch_size={batch_sizes[min_loss_index[1]]}, learning_rate={learning_rates[min_loss_index[2]]}, momentum={momentums[min_loss_index[3]]}, loss={min_loss_value}"
    )

    index = pd.MultiIndex.from_product(
        [epochs, batch_sizes, learning_rates, momentums],
        names=[
            "epochs",
            "batch_size",
            "learning_rate",
            "momentum",
        ],
    )
else:
    print(
        f"\nBest parameters for {args.dataset} (ADAM): epochs={epochs[min_loss_index[0]]}, batch_size={batch_sizes[min_loss_index[1]]}, learning_rate={learning_rates[min_loss_index[2]]}, loss={min_loss_value}"
    )

    index = pd.MultiIndex.from_product(
        [epochs, batch_sizes, learning_rates],
        names=["epochs", "batch_size", "learning_rate"],
    )

df = pd.DataFrame(avg_losses.flatten(), index=index, columns=["Average Loss"])
print(df.reset_index().to_string(index=False))
