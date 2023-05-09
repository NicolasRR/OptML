import threading
import torch
import torch.distributed.rpc as rpc
import argparse
from tqdm import tqdm
import numpy as np
from common import (
    create_worker_trainloaders,
    _get_model,
    get_optimizer,
    get_scheduler,
    get_model_accuracy,
    get_worker_accuracy,
    _save_model,
    save_weights,
    compute_weights_l2_norm,
    read_parser,
    start,
    LOSS_FUNC,
)


#################################### PARAMETER SERVER ####################################
class ParameterServer(object):
    def __init__(
        self,
        nb_workers,
        logger,
        dataset_name,
        learning_rate,
        momentum,
        use_alr,
        len_trainloader,
        epochs,
        lrs,
        saves_per_epoch,
    ):
        self.model = _get_model(dataset_name, LOSS_FUNC)
        self.logger = logger
        self.model_lock = threading.Lock()
        self.nb_workers = nb_workers
        self.loss = 0  # store workers loss
        self.optimizer = get_optimizer(self.model, learning_rate, momentum, use_alr)
        self.scheduler = get_scheduler(lrs, self.optimizer, len_trainloader, epochs)
        self.weights_matrix = []
        self.saves_per_epoch = saves_per_epoch
        if saves_per_epoch is not None:
            save_idx = np.linspace(0, len_trainloader - 1, saves_per_epoch, dtype=int)
            unique_idx = set(save_idx)
            if len(unique_idx) < saves_per_epoch:
                save_idx = np.array(sorted(unique_idx))
            self.save_idx = save_idx
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)

    def get_model(self):
        return self.model

    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(
        ps_rref,
        grads,
        worker_name,
        worker_batch_count,
        worker_epoch,
        total_batches_to_run,
        total_epochs,
        loss,
    ):
        self = ps_rref.local_value()
        self.logger.debug(
            f"PS got update from {worker_name}, {worker_batch_count - total_batches_to_run*(worker_epoch-1)}/{total_batches_to_run} ({worker_batch_count}/{total_batches_to_run*total_epochs}), epoch {worker_epoch}/{total_epochs}"
        )

        with self.model_lock:
            self.loss = loss

            for param, grad in zip(self.model.parameters(), grads):
                param.grad = grad

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.saves_per_epoch is not None:
                relative_batch_idx = (
                    worker_batch_count - total_batches_to_run * (worker_epoch - 1) - 1
                )
                if relative_batch_idx in self.save_idx:
                    weights = [
                        w.detach().clone().cpu().numpy()
                        for w in self.model.parameters()
                    ]
                    self.weights_matrix.append(weights)
            if worker_batch_count == total_batches_to_run:
                if self.scheduler is not None:
                    self.scheduler.step()

            self.logger.debug(
                f"PS updated model, worker loss: {loss} ({worker_name}), weight norm: weights norm {compute_weights_l2_norm(self.model)}"
            )

        return self.model


#################################### WORKER ####################################
class Worker(object):
    def __init__(self, ps_rref, logger, train_loader, epochs, worker_accuracy):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_func = LOSS_FUNC
        self.logger = logger
        self.batch_count = 0
        self.current_epoch = 0
        self.epochs = epochs
        self.worker_name = rpc.get_worker_info().name
        self.worker_accuracy = worker_accuracy
        self.logger.debug(
            f"{self.worker_name} is working on a dataset of size {len(train_loader.sampler)}"
        )
        self.progress_bar = tqdm(
            position=int(self.worker_name.split("_")[1]) - 1,
            desc=f"{self.worker_name}",
            unit="batch",
            total=len(self.train_loader) * self.epochs,
            leave=True,
        )

    def get_next_batch(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            current_lr = self.ps_rref.rpc_sync().get_current_lr()
            self.progress_bar.set_postfix(
                epoch=f"{self.current_epoch}/{self.epochs}", lr=f"{current_lr:.5f}"
            )
            for inputs, labels in self.train_loader:
                yield inputs, labels
        self.progress_bar.clear()
        self.progress_bar.close()

    def train(self):
        worker_model = self.ps_rref.rpc_sync().get_model()

        for inputs, labels in self.get_next_batch():
            loss = self.loss_func(worker_model(inputs), labels)
            loss.backward()
            self.batch_count += 1
            # in asynchronous we send the parameters to the server asynchronously and then we update the worker model synchronously
            rpc.rpc_async(
                self.ps_rref.owner(),
                ParameterServer.update_and_fetch_model,
                args=(
                    self.ps_rref,
                    [param.grad for param in worker_model.parameters()],
                    self.worker_name,
                    self.batch_count,
                    self.current_epoch,
                    len(self.train_loader),
                    self.epochs,
                    loss.detach(),
                ),
            )
            worker_model = self.ps_rref.rpc_sync().get_model()

            self.progress_bar.update(1)

            if self.worker_accuracy:
                if (
                    self.batch_count == len(self.train_loader)
                    and self.current_epoch == self.epochs
                ):
                    get_worker_accuracy(
                        worker_model, self.worker_name, self.train_loader
                    )


#################################### GLOBAL FUNCTIONS ####################################
def run_worker(ps_rref, logger, train_loader, epochs, worker_accuracy):
    worker = Worker(ps_rref, logger, train_loader, epochs, worker_accuracy)
    worker.train()


def run_parameter_server_async(
    workers,
    logger,
    dataset_name,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    learning_rate,
    momentum,
    train_split,
    batch_size,
    epochs,
    worker_accuracy,
    model_accuracy,
    save_model,
    subfolder,
    use_alr,
    saves_per_epoch,
    lrs,
):
    train_loaders, batch_size = create_worker_trainloaders(
        dataset_name,
        train_split,
        batch_size,
        model_accuracy,
        len(workers),
        split_dataset,
        split_labels,
        split_labels_unscaled,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    ps_rref = rpc.RRef(
        ParameterServer(
            len(workers),
            logger,
            dataset_name,
            learning_rate,
            momentum,
            use_alr,
            len(train_loaders),
            epochs,
            lrs,
            saves_per_epoch,
        )
    )
    futs = []

    if (
        not split_dataset and not split_labels and not split_labels_unscaled
    ):  # workers sharing samples
        logger.info(f"Starting asynchronous SGD training with {len(workers)} workers")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker,
                    args=(ps_rref, logger, train_loaders, epochs, worker_accuracy),
                )
            )

    else:
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker,
                    args=(ps_rref, logger, train_loaders[idx], epochs, worker_accuracy),
                )
            )

    torch.futures.wait_all(futs)

    logger.info("Finished training")
    print(f"Final train loss: {ps_rref.to_here().loss}")

    if model_accuracy:
        get_model_accuracy(ps_rref.to_here().model, train_loader_full)

    if save_model:
        _save_model(
            "async",
            dataset_name,
            ps_rref.to_here().model,
            len(workers),
            train_split,
            learning_rate,
            momentum,
            batch_size,
            epochs,
            subfolder,
            split_dataset,
            split_labels,
            split_labels_unscaled,
        )

    if saves_per_epoch is not None:
        save_weights(
            ps_rref.to_here().weights_matrix,
            "async",
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
    parser = argparse.ArgumentParser(
        description="Asynchronous Parallel SGD parameter-Server RPC based training"
    )
    args = read_parser(parser, "async")

    start(args, "async")
