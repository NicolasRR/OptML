import os
import torch
import torch.nn as nn
from torch import optim
import argparse
from tqdm import tqdm
import logging
import logging.handlers
import numpy as np
from helpers import CNN_CIFAR10, CNN_CIFAR100, CNN_MNIST, create_worker_trainloaders

DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 614310


#################################### LOGGER ####################################
def setup_logger(subfolder):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        fh = logging.FileHandler(os.path.join(subfolder, "log.log"), mode="w")
    else:
        fh = logging.FileHandler("log.log", mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def run(
    dataset_name,
    learning_rate,
    momentum,
    train_split,
    batch_size,
    epochs,
    model_accuracy,
    save_model,
    subfolder,
):
    if "mnist" in dataset_name:
        print("Created MNIST CNN")
        model = CNN_MNIST()  # global model
    elif "cifar100" in dataset_name:
        print("Created CIFAR100 CNN")
        model = CNN_CIFAR100()
    elif "cifar10" in dataset_name:
        print("Created CIFAR10 CNN")
        model = CNN_CIFAR10()
    else:
        print("Unknown dataset, cannot create CNN")
        exit()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    loss_func = nn.functional.nll_loss

    train_loaders, batch_size = create_worker_trainloaders(
        1,
        dataset_name,
        False,
        False,
        False,  # compatibility for async split_labels_unscaled
        train_split,
        batch_size,
        model_accuracy,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    logger = setup_logger(subfolder)
    logger.info("Start non distributed SGD training")

    last_loss = None

    for epoch in range(epochs):
        progress_bar = tqdm(
            total=len(train_loaders),
            unit="batch",
        )
        progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}")
        for batch_idx, (data, target) in enumerate(train_loaders):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            logger.debug(
                f"Loss: {loss.item()}, batch: {batch_idx+1}/{len(train_loaders)} ({batch_idx+1 + len(train_loaders)*epoch}/{len(train_loaders)*epochs}), epoch: {epoch+1}/{epochs}"
            )
            progress_bar.update(1)
        progress_bar.close()

    last_loss = loss.item()

    progress_bar.close()

    logger.info("Finished training")
    print(f"Final train loss: {last_loss}")

    if model_accuracy:
        correct_predictions = 0
        total_predictions = 0
        # memory efficient way (for large datasets)
        with torch.no_grad():  # No need to track gradients for evaluation
            for _, (data, target) in enumerate(train_loader_full):
                logits = model(data)
                predicted_classes = torch.argmax(logits, dim=1)
                correct_predictions += (predicted_classes == target).sum().item()
                total_predictions += target.size(0)
        final_train_accuracy = correct_predictions / total_predictions
        print(
            f"Final train accuracy: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})"
        )

    if save_model:
        filename = f"{dataset_name}_classic_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}.pt"

        if len(subfolder) > 0:
            filepath = os.path.join(subfolder, filename)
        else:
            filepath = filename

        torch.save(model.state_dict(), filepath)
        print(f"Model saved: {filepath}")


#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic non distributed SGD training")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        required=True,
        help="Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100.",
    )
    parser.add_argument(
        "--split_dataset",
        action="store_true",
        help="""After applying train_split, each worker will train on a unique distinct dataset (samples will not be 
        shared between workers).""",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=None,
        help="""Fraction of the training dataset to be used for training (0,1].""",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="""Learning rate of SGD  (0,+inf)."""
    )
    parser.add_argument(
        "--momentum", type=float, default=None, help="""Momentum of SGD  [0,+inf)."""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="""Number of epochs for training [1,+inf).""",
    )
    parser.add_argument(
        "--model_accuracy",
        action="store_true",
        help="""If set, will compute the train accuracy of the global model after training.""",
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="""If set, the trained model will not be saved.""",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="""If set, it will set seeds on torch, numpy and random for reproducibility purposes.""",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="""Subfolder where the model and log.log will be saved.""",
    )

    args = parser.parse_args()

    if args.train_split is None:
        args.train_split = DEFAULT_TRAIN_SPLIT
        print(f"Using default train_split value: {DEFAULT_TRAIN_SPLIT}")
    elif args.train_split > 1 or args.train_split <= 0:
        print("Forbidden value !!! train_split must be between (0,1]")
        exit()

    if args.lr is None:
        args.lr = DEFAULT_LR
        print(f"Using default lr: {DEFAULT_LR}")
    elif args.lr <= 0:
        print("Forbidden value !!! lr must be between (0,+inf)")
        exit()

    if args.momentum is None:
        args.momentum = DEFAULT_MOMENTUM
        print(f"Using default momentum: {DEFAULT_MOMENTUM}")
    elif args.momentum < 0:
        print("Forbidden value !!! momentum must be between [0,+inf)")
        exit()

    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS
        print(f"Using default epochs: {DEFAULT_EPOCHS}")
    elif args.epochs < 1:
        print("Forbidden value !!! epochs must be between [1,+inf)")
        exit()

    if args.seed:
        torch.manual_seed(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)

    if len(args.subfolder) > 0:
        print(f"Saving model and log.log to {args.subfolder}")

    run(
        args.dataset,
        args.lr,
        args.momentum,
        args.train_split,
        args.batch_size,
        args.epochs,
        args.model_accuracy,
        not args.no_save_model,
        args.subfolder,
    )
