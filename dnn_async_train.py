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
    compute_accuracy_loss,
    get_base_name,
    _save_model,
    save_weights,
    compute_weights_l2_norm,
    read_parser,
    start,
    _delay,
    LOSS_FUNC,
)

#################################### PARAMETER SERVER ####################################
class ParameterServer_async(object):
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
        val,
        alt_model,
        train_loader=None,
        val_loader=None,
    ):
        self.model = _get_model(dataset_name, LOSS_FUNC, alt_model)
        self.logger = logger
        self.model_lock = threading.Lock()
        self.nb_workers = nb_workers
        self.loss = 0  # store workers loss
        self.optimizer = get_optimizer(self.model, learning_rate, momentum, use_alr)
        self.scheduler = get_scheduler(lrs, self.optimizer, len_trainloader, epochs)
        self.weights_matrix = []
        self.saves_per_epoch = saves_per_epoch
        if lrs is not None or saves_per_epoch is not None or val:
            self.global_batch_counter = 0
        if saves_per_epoch is not None:
            save_idx = np.linspace(0, len_trainloader - 1, saves_per_epoch, dtype=int)
            unique_idx = set(save_idx)
            if len(unique_idx) < saves_per_epoch:
                save_idx = np.array(sorted(unique_idx))
            self.save_idx = save_idx
        self.val = val
        if val:
            self.train_loader = train_loader
            self.val_loader = val_loader
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)

    def get_model_async(self):
        return self.model

    def get_current_lr_async(self):
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model_async(
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
            if (
                self.scheduler is not None
                or self.saves_per_epoch is not None
                or self.val
            ):
                self.global_batch_counter += 1
            for param, grad in zip(self.model.parameters(), grads):
                param.grad = grad

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.saves_per_epoch is not None:
                relative_batch_idx = (
                    self.global_batch_counter / self.nb_workers - 1
                ) % total_batches_to_run
                if relative_batch_idx in self.save_idx:
                    weights = np.concatenate(
                        [
                            w.detach().clone().cpu().numpy().ravel()
                            for w in self.model.state_dict().values()
                        ]
                    )
                    self.weights_matrix.append(weights)

            if self.scheduler is not None or self.val:
                if (
                    self.global_batch_counter % (total_batches_to_run * self.nb_workers)
                    == 0
                ):
                    if self.scheduler is not None:
                        self.scheduler.step()
                    if self.val:
                        (
                            train_acc,
                            train_corr,
                            train_tot,
                            train_loss,
                        ) = compute_accuracy_loss(
                            self.model, self.train_loader, LOSS_FUNC, return_loss=True
                        )
                        val_acc, val_corr, val_tot, val_loss = compute_accuracy_loss(
                            self.model, self.val_loader, LOSS_FUNC, return_loss=True
                        )
                        self.logger.debug(
                            f"Train loss: {train_loss}, train accuracy: {train_acc*100} % ({train_corr}/{train_tot}), val loss: {val_loss}, val accuracy: {val_acc*100} % ({val_corr}/{val_tot}), epoch: {worker_epoch}/{total_epochs}"
                        )

            self.logger.debug(
                f"PS updated model, worker loss: {loss} ({worker_name}), weight norm: weights norm {compute_weights_l2_norm(self.model)}"
            )

        return self.model

#################################### WORKER ####################################
class Worker_async(object):
    def __init__(
        self,
        ps_rref,
        logger,
        train_loader,
        epochs,
        worker_accuracy,
        delay,
        delay_intensity,
        delay_type,
        slow_worker_1,
        dataset_name,
    ):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_func = LOSS_FUNC
        self.logger = logger
        self.batch_count = 0
        self.current_epoch = 0
        self.epochs = epochs
        self.worker_name = rpc.get_worker_info().name
        self.worker_accuracy = worker_accuracy
        self.delay = delay
        self.delay_intensity = delay_intensity
        self.delay_type = delay_type
        self.slow_worker_1 = slow_worker_1
        self.dataset_name = dataset_name
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

    def get_next_batch_async(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            current_lr = self.ps_rref.rpc_sync().get_current_lr_async()
            self.progress_bar.set_postfix(
                epoch=f"{self.current_epoch}/{self.epochs}", lr=f"{current_lr:.5f}"
            )
            for inputs, labels in self.train_loader:
                yield inputs, labels
        self.progress_bar.clear()
        self.progress_bar.close()

    def train_async(self):
        worker_model = self.ps_rref.rpc_sync().get_model_async()

        for inputs, labels in self.get_next_batch_async():
            loss = self.loss_func(worker_model(inputs), labels)
            loss.backward()
            self.batch_count += 1

            if self.worker_name == "Worker_1" and self.slow_worker_1:
                _delay(
                    intensity=self.delay_intensity, _type=self.delay_type, worker_1=True
                )
            elif self.delay:
                _delay(
                    intensity=self.delay_intensity,
                    _type=self.delay_type,
                    worker_1=False,
                )

            # in asynchronous we send the parameters to the server asynchronously and then we update the worker model synchronously
            rpc.rpc_async(
                self.ps_rref.owner(),
                ParameterServer_async.update_and_fetch_model_async,
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
            worker_model = self.ps_rref.rpc_sync().get_model_async()

            self.progress_bar.update(1)

        if self.worker_accuracy:
            (
                final_train_accuracy,
                correct_predictions,
                total_preidctions,
            ) = compute_accuracy_loss(
                worker_model,
                self.train_loader,
                loss_func=LOSS_FUNC,
                worker_mode=True,
                dataset_name=self.dataset_name,
                worker_name=self.worker_name,
            )
            print(
                f"Accuracy of {self.worker_name}: {final_train_accuracy*100} % ({correct_predictions}/{total_preidctions})"
            )

#################################### GLOBAL FUNCTIONS ####################################
def run_worker_async(
    ps_rref,
    logger,
    train_loader,
    epochs,
    worker_accuracy,
    delay,
    delay_intensity,
    delay_type,
    slow_worker_1,
    dataset_name=None,
):
    worker = Worker_async(
        ps_rref,
        logger,
        train_loader,
        epochs,
        worker_accuracy,
        delay,
        delay_intensity,
        delay_type,
        slow_worker_1,
        dataset_name,
    )
    worker.train_async()

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
    delay,
    delay_intensity,
    delay_type,
    slow_worker_1,
    val,
    alt_model,
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
        validation=val,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    if val:
        train_loader = train_loaders[0]
        val_loader = train_loaders[1]
        ps_rref = rpc.RRef(
            ParameterServer_async(
                len(workers),
                logger,
                dataset_name,
                learning_rate,
                momentum,
                use_alr,
                len(train_loader),
                epochs,
                lrs,
                saves_per_epoch,
                val,
                alt_model,
                train_loader=train_loader,
                val_loader=val_loader,
            )
        )
    else:
        train_loader = train_loaders
        if split_dataset or split_labels:
            len_train_loader = len(train_loader[0])
        elif split_labels_unscaled:
            len_train_loader = np.ceil(
                len(train_loader[0].dataset) / len(workers) / batch_size
            )
        else:
            len_train_loader = len(train_loader)
        ps_rref = rpc.RRef(
            ParameterServer_async(
                len(workers),
                logger,
                dataset_name,
                learning_rate,
                momentum,
                use_alr,
                len_train_loader,
                epochs,
                lrs,
                saves_per_epoch,
                val,
                alt_model,
            )
        )

    futs = []
    logger.info(f"Starting asynchronous SGD training with {len(workers)} workers")
    if (
        not split_dataset and not split_labels and not split_labels_unscaled
    ):  # workers sharing samples
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker_async,
                    args=(
                        ps_rref,
                        logger,
                        train_loader,
                        epochs,
                        worker_accuracy,
                        delay,
                        delay_intensity,
                        delay_type,
                        slow_worker_1,
                    ),
                )
            )

    else:
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker_async,
                    args=(
                        ps_rref,
                        logger,
                        train_loader[idx],
                        epochs,
                        worker_accuracy,
                        delay,
                        delay_intensity,
                        delay_type,
                        slow_worker_1,
                        dataset_name,
                    ),
                )
            )

    torch.futures.wait_all(futs)

    logger.info("Finished training")
    print(f"Final train loss: {ps_rref.to_here().loss}")

    if model_accuracy:
        (
            final_train_accuracy,
            correct_predictions,
            total_preidctions,
        ) = compute_accuracy_loss(ps_rref.to_here().model, train_loader_full, LOSS_FUNC)
        print(
            f"Final train accuracy: {final_train_accuracy*100} % ({correct_predictions}/{total_preidctions})"
        )

    base_name = get_base_name(
        "async",
        dataset_name,
        len(workers) + 1,
        train_split,
        learning_rate,
        momentum,
        batch_size,
        epochs,
        val,
        use_alr,
        lrs,
        saves_per_epoch,
        alt_model=alt_model,
        split_dataset=split_dataset,
        split_labels=split_labels,
        split_labels_unscaled=split_labels_unscaled,
        delay=delay,
        slow_worker_1=slow_worker_1,
        delay_intensity=delay_intensity,
        delay_type=delay_type,
    )

    if save_model:
        _save_model(
            base_name,
            subfolder,
            ps_rref.to_here().model,
        )

    if saves_per_epoch is not None:
        save_weights(
            base_name,
            subfolder,
            ps_rref.to_here().weights_matrix,
        )

#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asynchronous Parallel SGD parameter-Server RPC based training"
    )
    args = read_parser(parser, "async")

    start(args, "async", run_parameter_server_async)


