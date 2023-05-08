import argparse
from tqdm import tqdm
import numpy as np
from common import (
    create_worker_trainloaders,
    setup_simple_logger,
    _get_model,
    get_optimizer,
    get_scheduler,
    get_model_accuracy,
    _save_model,
    save_weights,
    compute_weights_l2_norm,
    read_parser,
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
    args = read_parser(parser)

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
