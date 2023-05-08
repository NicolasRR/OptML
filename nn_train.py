import torch
import argparse
from tqdm import tqdm
import numpy as np
from helpers import (
    create_worker_trainloaders,
    setup_simple_logger,
    _get_model,
    get_optimizer,
    get_scheduler,
    get_model_accuracy,
    _save_model,
    save_weights,
    compute_weights_l2_norm,
    DEFAULT_DATASET,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_EPOCHS,
    DEFAULT_SEED,
    LOSS_FUNC,
)


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

    model = _get_model(dataset_name, loss_func)
    optimizer = get_optimizer(model, learning_rate, momentum, use_alr)

    train_loaders, batch_size = create_worker_trainloaders(
        dataset_name,
        train_split,
        batch_size,
        model_accuracy,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    scheduler = get_scheduler(lrs, optimizer, len(train_loaders), epochs)

    logger = setup_simple_logger(subfolder)
    logger.info("Start non distributed SGD training")

    if saves_per_epoch is not None:
        weights_matrix = []
        save_idx = np.linspace(0, len(train_loaders) - 1, saves_per_epoch, dtype=int)
        unique_idx = set(save_idx)
        if len(unique_idx) < saves_per_epoch:
            save_idx = np.array(sorted(unique_idx))

    last_loss = None

    for epoch in range(epochs):
        progress_bar = tqdm(
            total=len(train_loaders),
            unit="batch",
        )
        progress_bar.set_postfix(
            epoch=f"{epoch+1}/{epochs}", lr=f"{optimizer.param_groups[0]['lr']:.5f}"
        )
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
                    weights = [
                        w.detach().clone().cpu().numpy() for w in model.parameters()
                    ]
                    weights_matrix.append(weights)

            progress_bar.update(1)
        progress_bar.close()
        if scheduler is not None:
            scheduler.step()

    last_loss = loss.item()

    progress_bar.close()

    logger.info("Finished training")
    print(f"Final train loss: {last_loss}")

    if model_accuracy:
        get_model_accuracy(model, train_loader_full)

    if save_model:
        _save_model(
            "classic",
            dataset_name,
            model,
            -1,
            train_split,
            learning_rate,
            momentum,
            batch_size,
            epochs,
            subfolder,
        )

    if saves_per_epoch is not None:
        save_weights(
            weights_matrix,
            "classic",
            dataset_name,
            train_split,
            learning_rate,
            momentum,
            batch_size,
            epochs,
            subfolder,
        )


#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic non distributed SGD training")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
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
        help="""Number of times the model weights will be saved during one epoch.""",
    )
    parser.add_argument(
        "--alr",
        action="store_true",
        help="""If set, use adaptive learning rate (Adam optimizer) instead of SGD optimizer.""",
    )
    parser.add_argument(
        "--lrs",
        type=str,
        choices=["exponential", "cosine_annealing"],
        default=None,
        help="""Choose a learning rate scheduler: exponential, cosine_annealing, or none.""",
    )

    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = DEFAULT_DATASET
        print(f"Using default dataset: {DEFAULT_DATASET}")

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
