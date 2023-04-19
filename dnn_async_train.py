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
from helpers import CNN_CIFAR10, CNN_CIFAR100, CNN_MNIST, create_worker_trainloaders

DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 614310


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


def log_writer(log_queue):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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
        self.loss = np.array([])  # store workers loss
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

        self.loss = np.append(self.loss, loss)

        with self.model_lock:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=False)
            fut = torch.futures.Future()
            fut.set_result(self.model)
            self.logger.debug(f"PS updated model")

        return fut


#################################### WORKER ####################################
class Worker(object):
    def __init__(self, ps_rref, logger, train_loader, epochs, worker_accuracy, tqdm_lock):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_func = nn.functional.nll_loss  # worker loss function
        self.logger = logger
        self.batch_count = 0
        self.current_epoch = 0
        self.epochs = epochs
        self.worker_name = rpc.get_worker_info().name
        self.worker_accuracy = worker_accuracy
        self.tqdm_lock = tqdm_lock
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
                    self.worker_name,
                    self.batch_count,
                    self.current_epoch,
                    len(self.train_loader),
                    self.epochs,
                    loss.detach(),
                ),
            )
            worker_model = self.ps_rref.rpc_sync().get_model()
            with self.tqdm_lock:
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
def run_worker(ps_rref, logger, train_loader, epochs, worker_accuracy, tqdm_lock):
    worker = Worker(ps_rref, logger, train_loader, epochs, worker_accuracy, tqdm_lock)
    worker.train()


def run_parameter_server(
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
    train_loader_full = 0
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    ps_rref = rpc.RRef(
        ParameterServer(len(workers), logger, dataset_name, learning_rate, momentum)
    )
    futs = []

    tqdm_lock = threading.Lock()

    if (
        not split_dataset and not split_labels and not split_labels_unscaled
    ):  # workers sharing samples
        logger.info("Starting asynchronous SGD training")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker,
                    args=(ps_rref, logger, train_loaders, epochs, worker_accuracy, tqdm_lock),
                )
            )

    else:
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(
                    worker,
                    run_worker,
                    args=(ps_rref, logger, train_loaders[idx], epochs, worker_accuracy, tqdm_lock),
                )
            )

    torch.futures.wait_all(futs)

    loss = ps_rref.to_here().loss

    logger.info("Finished training")
    print(f"Final train loss: {loss[-1]}")

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
        if split_dataset:
            filename = f"{dataset_name}_async_{len(workers)+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}_split_dataset.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")
        elif split_labels:
            filename = f"{dataset_name}_async_{len(workers)+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}_labels.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")
        elif split_labels_unscaled:
            filename = f"{dataset_name}_async_{len(workers)+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}_labels_unscaled.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")
        else:
            filename = f"{dataset_name}_async_{len(workers)+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")


def run(
    rank,
    log_queue,
    dataset_name,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    world_size,
    learning_rate,
    momentum,
    train_split,
    batch_size,
    epochs,
    worker_accuracy,
    model_accuracy,
    save_model,
):
    logger = setup_logger(log_queue)
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=world_size, rpc_timeout=0  # infinite timeout
    )

    if rank != 0:
        # starting up worker
        rpc.init_rpc(
            f"Worker_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # worker passively waiting for parameter server to kick off training iterations
    else:
        # parameter server gives data to the workers
        rpc.init_rpc(
            "Parameter_Server",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        run_parameter_server(
            [f"Worker_{r}" for r in range(1, world_size)],
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
        )

    # block until all rpcs finish
    rpc.shutdown()


#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asynchronous Parallel SGD parameter-Server RPC based training"
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        required=True,
        help="Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes [2,+inf].""",
    )
    parser.add_argument(
        "--split_dataset",
        action="store_true",
        help="""After applying train_split, each worker will train on a unique distinct dataset (samples will not be 
        shared between workers).""",
    )
    parser.add_argument(
        "--split_labels",
        action="store_true",
        help="""If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. 
        Workers will not share samples and the labels are randomly assigned. Note, the training length will be the SAME for all workers (like in synchronous SGD).
        This mode requires --batch_size 1, don't use --split_dataset. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
    )
    parser.add_argument(
        "--split_labels_unscaled",
        action="store_true",
        help="""If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. 
        Workers will not share samples and the labels are randomly assigned. Note, the training length will be the DIFFERENT for all workers, based on the number of samples each class has.
        This mode requires --batch_size 1, don't use --split_dataset. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=None,
        help="""Percentage of the training dataset to be used for training (0,1].""",
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
        "--worker_accuracy",
        action="store_true",
        help="""If set, will compute the train accuracy of each worker after training (useful when --split_dataset).""",
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

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if args.world_size is None:
        args.world_size = DEFAULT_WORLD_SIZE
        print(f"Using default world_size value: {DEFAULT_WORLD_SIZE}")
    elif args.world_size < 2:
        print(
            "Forbidden value !!! world_size must be >= 2 (1 Parameter Server and 1 Worker)"
        )
        exit()

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

    if args.no_save_model:
        save_model = False
    else:
        save_model = True

    if args.split_dataset:
        split_dataset = True
    else:
        split_dataset = False

    if args.model_accuracy:
        model_accuracy = True
    else:
        model_accuracy = False

    if args.worker_accuracy:
        worker_accuracy = True
    else:
        worker_accuracy = False

    if args.split_labels:
        split_labels = True
    else:
        split_labels = False

    if args.split_labels_unscaled:
        split_labels_unscaled = True
    else:
        split_labels_unscaled = False

    if split_labels:
        if split_dataset:
            print("Please use --split_labels without --split_dataset")
            exit()
        elif args.batch_size != 1:
            print("Please use --split_labels with the --batch_size 1")
            exit()
        elif split_labels_unscaled:
            print("Please use --split_labels with the --split_labels_unscaled")
            exit()
        elif args.dataset == "mnist":
            if 10 % (args.world_size - 1) != 0 or args.world_size == 2:
                print("Please use --split_labels with --world_size {3, 6, 11}")
                exit()
        elif args.dataset == "fashion_mnist":
            if 40 % (args.world_size - 1) != 0 or args.world_size == 2:
                print(
                    "Please use --split_labels with --world_size {3, 5, 6, 9, 11, 21, 41}"
                )
                exit()
        elif args.dataset == "cifar10":
            if 10 % (args.world_size - 1) != 0 or args.world_size == 2:
                print("Please use --split_labels with --world_size {3, 6, 11}")
                exit()
        elif args.dataset == "cifar100":
            if 100 % (args.world_size - 1) != 0 or args.world_size == 2:
                print(
                    "Please use --split_labels with --world_size {3, 5, 6, 11, 21, 26, 51, 101}"
                )
                exit()

    if split_labels_unscaled:
        if split_dataset:
            print("Please use --split_labels_unscaled without --split_dataset")
            exit()
        elif args.batch_size != 1:
            print("Please use --split_labels_unscaled with the --batch_size 1")
            exit()
        elif split_labels:
            print("Please use --split_labels_unscaled with the --split_labels")
            exit()
        elif args.dataset == "mnist":
            if 10 % (args.world_size - 1) != 0 or args.world_size == 2:
                print("Please use --split_labels_unscaled with --world_size {3, 6, 11}")
                exit()
        elif args.dataset == "fashion_mnist":
            if 40 % (args.world_size - 1) != 0 or args.world_size == 2:
                print(
                    "Please use --split_labels_unscaled with --world_size {3, 5, 6, 9, 11, 21, 41}"
                )
                exit()
        elif args.dataset == "cifar10":
            if 10 % (args.world_size - 1) != 0 or args.world_size == 2:
                print("Please use --split_labels_unscaled with --world_size {3, 6, 11}")
                exit()
        elif args.dataset == "cifar100":
            if 100 % (args.world_size - 1) != 0 or args.world_size == 2:
                print(
                    "Please use --split_labels_unscaled with --world_size {3, 5, 6, 11, 21, 26, 51, 101}"
                )
                exit()

    if args.seed:
        torch.manual_seed(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)

    with Manager() as manager:
        log_queue = manager.Queue()
        log_writer_thread = threading.Thread(target=log_writer, args=(log_queue,))
        log_writer_thread.start()

        mp.spawn(
            run,
            args=(
                log_queue,
                args.dataset,
                split_dataset,
                split_labels,
                split_labels_unscaled,
                args.world_size,
                args.lr,
                args.momentum,
                args.train_split,
                args.batch_size,
                args.epochs,
                worker_accuracy,
                model_accuracy,
                save_model,
            ),
            nprocs=args.world_size,
            join=True,
        )

        log_queue.put(None)
        log_writer_thread.join()
