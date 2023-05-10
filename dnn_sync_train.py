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
    _save_model,
    save_weights,
    compute_weights_l2_norm,
    read_parser,
    start,
    long_random_delay,
    random_delay,
    LOSS_FUNC,
)


#################################### PARAMETER SERVER ####################################
class ParameterServer_sync(object):
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
        train_loader= None,
        val_loader=None,
    ):
        self.model = _get_model(dataset_name, LOSS_FUNC)
        self.logger = logger
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.nb_workers = nb_workers
        self.update_counter = 0
        self.model_loss = 0  # store global model loss (averaged workers loss)
        self.loss = np.array([])  # store workers loss
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
        self.val = val
        if val:
            print(f"setting lodaers {type(train_loader)}, {type(val_loader)}")
            self.train_loader = train_loader
            self.val_loader = val_loader
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)

    def get_model_sync(self):
        return self.model

    def get_current_lr_sync(self):
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model_sync(
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
            f"PS got {self.update_counter +1}/{self.nb_workers} updates (from {worker_name}, {worker_batch_count - total_batches_to_run*(worker_epoch-1)}/{total_batches_to_run} ({worker_batch_count}/{total_batches_to_run*total_epochs}), epoch {worker_epoch}/{total_epochs})"
        )
        for param, grad in zip(self.model.parameters(), grads):
            if (param.grad is not None) and (grad is not None):
                param.grad += grad  # accumulate workers grads

        self.loss = np.append(self.loss, loss)

        with self.lock:
            self.update_counter += 1
            fut = self.future_model

            if (
                self.update_counter >= self.nb_workers
            ):  # received grads from all workers
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= self.nb_workers  # average workers grads
                self.update_counter = 0
                self.model_loss = self.loss.mean()  # aggregate the workers loss
                self.loss = np.array([])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)  # reset grad tensor to 0
                if self.saves_per_epoch is not None:
                    relative_batch_idx = (
                        worker_batch_count
                        - total_batches_to_run * (worker_epoch - 1)
                        - 1
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
                    if self.val:
                        print(type(self.train_loader), type(self.val_loader))
                        train_acc, train_corr, train_tot, train_loss = compute_accuracy_loss(self.model, self.train_loader, LOSS_FUNC, return_loss=True)
                        val_acc, val_corr, val_tot, val_loss = compute_accuracy_loss(self.model, self.val_loader, LOSS_FUNC, return_loss=True)
                        self.logger.debug(
                                f"Train loss: {train_loss}, train accuracy: {train_acc*100} % ({train_corr}/{train_tot}), val loss: {val_loss}, val accuracy: {val_acc*100} % ({val_corr}/{val_tot}), epoch: {worker_epoch}/{total_epochs}"
                        )
                    
                fut.set_result(self.model)
                self.logger.debug(
                    f"PS updated model, global loss is {self.model_loss}, weights norm is {compute_weights_l2_norm(self.model)}"
                )
                self.future_model = torch.futures.Future()

        return fut


#################################### WORKER ####################################
class Worker_sync(object):
    def __init__(
        self,
        ps_rref,
        logger,
        train_loader,
        epochs,
        worker_accuracy,
        delay,
        slow_worker_1,
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
        self.slow_worker_1 = slow_worker_1
        self.logger.debug(
            f"{self.worker_name} is working on a dataset of size {len(train_loader.sampler)}"
        )

    def get_next_batch_sync(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            if self.worker_name == "Worker_1":
                # progress bar only of the first worker (we are in synchronous mode)
                iterable = tqdm(
                    self.train_loader,
                    unit="batch",
                )
                current_lr = self.ps_rref.rpc_sync().get_current_lr_sync()
                iterable.set_postfix(
                    epoch=f"{self.current_epoch}/{self.epochs}", lr=f"{current_lr:.5f}"
                )
            else:
                iterable = self.train_loader

            for inputs, labels in iterable:
                yield inputs, labels

        if self.worker_name == "Worker_1":
            iterable.close()

    def train_sync(self):
        worker_model = self.ps_rref.rpc_sync().get_model_sync()

        for inputs, labels in self.get_next_batch_sync():
            loss = self.loss_func(worker_model(inputs), labels)
            loss.backward()
            self.batch_count += 1

            if self.worker_name == "Worker_1" and self.slow_worker_1:
                long_random_delay()
            elif self.delay:
                random_delay()

            worker_model = rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer_sync.update_and_fetch_model_sync,
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
        if self.worker_accuracy:
            final_train_accuracy, correct_predictions, total_predictions = compute_accuracy_loss(worker_model, self.train_loader, loss_func=LOSS_FUNC)
            print(
                f"Accuracy of {self.worker_name}: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})" 
            )


#################################### GLOBAL FUNCTIONS ####################################
def run_worker_sync(
    ps_rref, logger, train_loader, epochs, worker_accuracy, delay, slow_worker_1
):
    worker = Worker_sync(
        ps_rref, logger, train_loader, epochs, worker_accuracy, delay, slow_worker_1
    )
    worker.train_sync()


def run_parameter_server_sync(
    workers,
    logger,
    dataset_name,
    split_dataset,
    split_labels,
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
    slow_worker_1,
    val,
):
    train_loaders, batch_size = create_worker_trainloaders(
        dataset_name,
        train_split,
        batch_size,
        model_accuracy,
        len(workers),
        split_dataset,
        split_labels,
        val,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]
    if val:
        print("val is true")
        train_loader = train_loaders[0]
        val_loader = train_loaders[1]
        print(type(train_loader), type(val_loader))
        ps_rref = rpc.RRef(
            ParameterServer_sync(
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
                train_loader=train_loader,
                val_loader=val_loader,
            )
        )
    else:
        train_loader = train_loaders
        ps_rref = rpc.RRef(
            ParameterServer_sync(
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
            )
        )
    futs = []

    if not split_dataset and not split_labels:  # workers sharing samples
        logger.info(f"Starting synchronous SGD training with {len(workers)} workers")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker_sync,
                    args=(
                        ps_rref,
                        logger,
                        train_loader,
                        epochs,
                        worker_accuracy,
                        delay,
                        slow_worker_1,
                    ),
                )
            )

    else:
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker_sync,
                    args=(
                        ps_rref,
                        logger,
                        train_loader[idx],
                        epochs,
                        worker_accuracy,
                        delay,
                        slow_worker_1,
                    ),
                )
            )

    torch.futures.wait_all(futs)

    logger.info("Finished training")
    print(f"Final train loss: {ps_rref.to_here().model_loss}")

    if model_accuracy:
        final_train_accuracy, correct_predictions, total_preidctions = compute_accuracy_loss(ps_rref.to_here().model, train_loader_full, LOSS_FUNC)
        print(
            f"Final train accuracy: {final_train_accuracy*100} % ({correct_predictions}/{total_preidctions})"
        )

    if save_model:
        _save_model(
            "sync",
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
        )

    if saves_per_epoch is not None:
        save_weights(
            ps_rref.to_here().weights_matrix,
            "sync",
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
        description="Synchronous Parallel SGD parameter-Server RPC based training"
    )
    args = read_parser(parser, "sync")

    start(args, "sync", run_parameter_server_sync)
