import os
import threading
import queue
import torch
import torch.nn as nn
from torch import optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from multiprocessing import Manager
import argparse
from tqdm import tqdm
import logging
import logging.handlers
import numpy as np
from common import (
    create_worker_trainloaders,
    log_writer,
    setup_logger,
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


#################################### LOGGER ####################################
def setup_logger(log_queue):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    qh = QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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


def log_writer(log_queue, subfolder):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        with open(os.path.join(subfolder, "log_async.log"), "w") as log_file:
            while True:
                try:
                    record = log_queue.get(timeout=1)
                    if record is None:
                        break
                    msg = formatter.format(record)
                    log_file.write(msg + "\n")
                except queue.Empty:
                    continue
    else:
        with open("log_async.log", "w") as log_file:
            while True:
                try:
                    record = log_queue.get(timeout=1)
                    if record is None:
                        break
                    msg = formatter.format(record)
                    log_file.write(msg + "\n")
                except queue.Empty:
                    continue


#################################### PARAMETER SERVER ####################################
class ParameterServer(object):
    def __init__(self, nb_workers, logger, dataset_name, learning_rate, momentum):
        if "mnist" in dataset_name:
            print("Created MNIST CNN")
            self.model = CNN_MNIST()  # global model
        elif "cifar100" in dataset_name:
            print("Created CIFAR100 CNN")
            self.model = CNN_CIFAR100()
        elif "cifar10" in dataset_name:
            print("Created CIFAR10 CNN")
            self.model = CNN_CIFAR10()
        else:
            print("Unknown dataset, cannot create CNN")
            exit()

        self.logger = logger
        self.model_lock = threading.Lock()
        self.nb_workers = nb_workers
        self.loss = 0  # store workers loss
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=learning_rate, momentum=momentum
        )
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)

    def get_model(self):
        return self.model  # get global model

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

            self.logger.debug(f"PS updated model, worker loss: {loss} ({worker_name})")

        return self.model


#################################### WORKER ####################################
class Worker(object):
    def __init__(self, ps_rref, logger, train_loader, epochs, worker_accuracy):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_func = nn.functional.nll_loss  # worker loss function
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
            self.progress_bar.set_postfix(epoch=f"{self.current_epoch}/{self.epochs}")
            for inputs, labels in self.train_loader:
                yield inputs, labels
        self.progress_bar.clear()
        self.progress_bar.close()

    def train(self):
        worker_model = self.ps_rref.rpc_sync().get_model()

        for inputs, labels in self.get_next_batch():
            loss = self.loss_func(worker_model(inputs), labels)  # worker loss
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
                    correct_predictions = 0
                    total_predictions = 0
                    with torch.no_grad():  # No need to track gradients for evaluation
                        for _, (data, target) in enumerate(self.train_loader):
                            logits = worker_model(data)
                            predicted_classes = torch.argmax(logits, dim=1)
                            correct_predictions += (
                                (predicted_classes == target).sum().item()
                            )
                            total_predictions += target.size(0)
                        final_train_accuracy = correct_predictions / total_predictions
                    print(
                        f"Accuracy of {self.worker_name}: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})"
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
):
    train_loaders, batch_size = create_worker_trainloaders(
        len(workers),
        dataset_name,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        train_split,
        batch_size,
        model_accuracy,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    ps_rref = rpc.RRef(
        ParameterServer(len(workers), logger, dataset_name, learning_rate, momentum)
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
        correct_predictions = 0
        total_predictions = 0
        # memory efficient way (for large datasets)
        with torch.no_grad():  # No need to track gradients for evaluation
            for _, (data, target) in enumerate(train_loader_full):
                logits = ps_rref.to_here().model(data)
                predicted_classes = torch.argmax(logits, dim=1)
                correct_predictions += (predicted_classes == target).sum().item()
                total_predictions += target.size(0)
        final_train_accuracy = correct_predictions / total_predictions
        print(
            f"Final train accuracy: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})"
        )

    if save_model:
        suffix = ""
        if split_dataset:
            suffix = "_split_dataset"
        elif split_labels:
            suffix = "_labels"
        elif split_labels_unscaled:
            suffix = "_labels_unscaled"

        filename = f"{dataset_name}_async_{len(workers)+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}{suffix}.pt"

        if len(subfolder) > 0:
            filepath = os.path.join(subfolder, filename)
        else:
            filepath = filename

        torch.save(ps_rref.to_here().model.state_dict(), filepath)
        print(f"Model saved: {filepath}")


#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asynchronous Parallel SGD parameter-Server RPC based training"
    )
    args = read_parser(parser, "async")

    start(args, "async")
