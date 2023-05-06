import os
import torch
from torch import optim
import argparse
from tqdm import tqdm
import numpy as np
from helpers import CNN_CIFAR10, CNN_CIFAR100, CNN_MNIST, create_worker_trainloaders, setup_simple_logger, compute_weights_l2_norm, LOSS_FUNC, EXPO_DECAY

DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 614310

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
    saves_per_epoch,
    use_alr,
    lrs,
):
    
    loss_func = LOSS_FUNC

    if "mnist" in dataset_name:
        print("Created MNIST CNN")
        model = CNN_MNIST(loss_func=loss_func)  # global model
    elif "cifar100" in dataset_name:
        print("Created CIFAR100 CNN")
        model = CNN_CIFAR100(loss_func=loss_func)
    elif "cifar10" in dataset_name:
        print("Created CIFAR10 CNN")
        model = CNN_CIFAR10(loss_func=loss_func)
    else:
        print("Unknown dataset, cannot create CNN")
        exit()

    if use_alr:
        if momentum > 1:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(max(momentum, 0.99), 0.999)) # weight decay if weights too large
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loaders, batch_size = create_worker_trainloaders(
        1, # only 1 worker
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

    if lrs is not None:
        if args.lrs == "exponential": # more suitable for async
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=EXPO_DECAY)  # initial_learning_rate * gamma^epoch
        elif args.lrs == "cosine_annealing": # more suitable for sync
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loaders) * epochs)
        else:
            scheduler = None

    logger = setup_simple_logger(subfolder)
    logger.info("Start non distributed SGD training")

    last_loss = None

    if saves_per_epoch is not None:
        weights_matrix = []
        save_idx = np.linspace(0, len(train_loaders) - 1, saves_per_epoch, dtype=int)
        unique_idx = set(save_idx)
        if len(unique_idx) < saves_per_epoch:
            save_idx = np.array(sorted(unique_idx))

    for epoch in range(epochs):
        progress_bar = tqdm(
            total=len(train_loaders),
            unit="batch",
        )
        progress_bar.set_postfix(epoch=f"{epoch+1}/{epochs}", lr=f"{optimizer.param_groups[0]['lr']:.5f}")
        for batch_idx, (data, target) in enumerate(train_loaders):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            logger.debug(
                f"Loss: {loss.item()}, weight norm: {compute_weights_l2_norm(model)}, batch: {batch_idx+1}/{len(train_loaders)} ({batch_idx+1 + len(train_loaders)*epoch}/{len(train_loaders)*epochs}), epoch: {epoch+1}/{epochs}"
            )
            if saves_per_epoch is not None:
                if batch_idx in save_idx:
                    weights = [w.detach().clone().cpu().numpy() for w in model.parameters()]
                    weights_matrix.append(weights)

            progress_bar.update(1)
        progress_bar.close()
        if scheduler is not None:
                scheduler.step()

    last_loss = loss.item()

    progress_bar.close()

    if saves_per_epoch is not None:
        flat_weights = [np.hstack([w.flatten() for w in epoch_weights]) for epoch_weights in weights_matrix]
        weights_matrix_np = np.vstack(flat_weights)

        filename = f"{dataset_name}_weights_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}.npy"
        if len(subfolder) > 0:
            filepath = os.path.join(subfolder, filename)
        else:
            filepath = filename

        np.save(filepath, weights_matrix_np)
        print(f"Weights saved: {filepath}")

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
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        help="Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100.",
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
    parser.add_argument(
    "--saves_per_epoch",
    type=int,
    default=None,
    help="Number of times the model weights will be saved during one epoch.",
    )
    parser.add_argument(
    "--alr",
    action="store_true",
    help="If set, use adaptive learning rate (Adam optimizer) instead of SGD optimizer.",
    )
    parser.add_argument(
    "--lrs",
    type=str,
    choices=["exponential", "cosine_annealing", "none"],
    default="none",
    help="Choose a learning rate scheduler: exponential, cosine_annealing, or none.",
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

    if args.saves_per_epoch is not None:
        if args.saves_per_epoch < 1:
            print("Forbidden value !!! saves_per_epoch must be > 1")
            exit()
        else:
            print(f"Saving model weights {args.saves_per_epoch} times during one epoch")

    if args.seed:
        torch.manual_seed(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)

    if len(args.subfolder) > 0:
        print(f"Saving model and log.log to {args.subfolder}")

    if args.alr:
        print("Using Adam as optimizer instead of SGD")

    if args.lrs is not None:
        print(f"Using learning rate scheduler: {args.lrs}")

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
        args.saves_per_epoch,
        args.alr,
        args.lrs,
    )
