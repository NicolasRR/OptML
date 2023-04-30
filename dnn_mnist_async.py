import os
import threading
from datetime import datetime
import time
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import torchvision
import argparse
from tqdm import tqdm
import logging
import logging.handlers
import numpy as np
import copy
from torch.optim.swa_utils import SWALR
import itertools
from multiprocessing import Manager
from helpers import CNN_MNIST, CNN_CIFAR100, CNN_CIFAR10,setup_logger, QueueHandler, log_writer
import queue


DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_BATCH_SIZE = 32 # 1 == SGD, >1 MINI BATCH SGD
DEFAULT_EPOCHS = 1



class ParameterServer(object):
    """
    This class contains the parameters and updates them at every iteration. To do so it performs thread locking in order to avoid memory issues.

    """

    def __init__(
        self,
        delay,
        ntrainers,
        lr,
        momentum,
        logger,
        mode,
        pca_gen,
        output_folder,
        dataset_name
    ):
        if dataset_name == "mnist" or dataset_name == "fmnist":
            self.model = CNN_MNIST()
        elif dataset_name == "cifar10":
            self.model = CNN_CIFAR10()
        elif dataset_name == "cifar100":
            self.model = CNN_CIFAR100()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        if mode == "scheduler" or mode == "swa":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=100 * lr, momentum=momentum
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum
            )
        self.losses = np.array([])
        self.norms = np.array([])
        for _, p in enumerate(self.model.parameters()):
            p.grad = torch.zeros_like(p)
        self.delay = delay
        self.logger = logger
        self.mode = mode
        if mode == "scheduler":
            self.logger.info("Creating a Learning rate scheduler")
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9
            )
        if mode == "swa":
            self.logger.info("Creating a SWA scheduler")
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9
            )
            self.swa_start = 2
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)
        elif mode == "compensation":
            self.logger.info("Using delay compensation")
            self.backups = [
                [
                    torch.randn_like(param, requires_grad=False)
                    for param in self.model.parameters()
                ]
                for _ in range(ntrainers)
            ]
        self.epochs = np.zeros(ntrainers, dtype=int)
        self.steps = np.zeros(ntrainers, dtype=int)
        # Uncomment these lines to generate the data for the PCA matrix generation
        self.pca_gen = pca_gen
        if pca_gen:
            self.filename = os.path.join(output_folder, "data_pca.npy")
            self.pca_current = 0
            self.pca_max = 300
            shape = (
                self.pca_max,
                torch.numel(
                    torch.nn.utils.parameters_to_vector(self.model.parameters())
                ),
            )
            self.data = np.lib.format.open_memmap(
                self.filename, mode="w+", shape=shape, dtype=np.float32
            )
            self.pca_iter = 3

    def get_model(self, id):
        if self.mode == "compensation":
            self.backups[id - 1] = [param for param in self.model.parameters()]
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, id, loss, epoch):
        """
        This function can be called asynchonously by all the slaves in order to update the master's parameters. It manually add the gradient to .grad
        """
        self = ps_rref.local_value()
        self.logger.debug(f"PS got 1 updates from worker {id}")

        self.losses = np.append(self.losses, loss)
        if id == 1:
            time.sleep(self.delay)

        with self.lock:
            fut = self.future_model
            norm = 0
            # PCA data generation
            if self.pca_gen and self.steps[id - 1] % self.pca_iter == 0:
                flat = np.array([])
            for i, (p, g) in enumerate(zip(self.model.parameters(), grads)):
                if p.grad is not None:
                    if self.mode == "compensation":
                        p.grad += g + 0.5 * g * g * (p - self.backups[id - 1][i])
                    else:
                        p.grad += g
                    if self.pca_gen and self.steps[id - 1] % self.pca_iter == 0:
                        flat = np.concatenate((flat, torch.flatten(p.grad).numpy()))
                    n = p.grad.norm(2).item() ** 2
                    norm += n
                else:
                    self.logger.debug(f"None p.grad detected for the update")
                    # Append the data to the memory-mapped array
            # PCA data generation
            if (
                self.pca_gen
                and self.pca_current < self.pca_max
                and self.steps[id - 1] % self.pca_iter == 0
            ):
                # self.data = np.vstack([self.data, flat])
                self.data[self.pca_current, :] = flat
                self.pca_current += 1
                if self.pca_current % 10 == 0:
                    self.data.flush()
                # np.save(self.filename, self.data)
            self.logger.debug(f"Loss is {self.losses[-1]}")
            self.norms = np.append(self.norms, norm ** (0.5))
            self.logger.debug(f"The norm of the gradient is {self.norms[-1]}")
            self.loss = np.array([])
            self.optimizer.step()
            self.steps[id - 1] += 1
            if self.epochs[id - 1] < epoch:
                self.steps[id - 1] = 0
                self.epochs[id - 1] += 1
                if self.mode == "swa" and epoch > self.swa_start:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                    self.logger.debug(
                        f"SWA scheduler update: {self.swa_scheduler.get_lr()}"
                    )
                elif self.mode == "swa" or self.mode == "scheduler":
                    self.scheduler.step()
                    lrs = [param["lr"] for param in self.optimizer.param_groups]
                    self.logger.debug(f"Scheduler Update: {lrs}")

            self.optimizer.zero_grad(set_to_none=False)
            fut.set_result(self.model)
            self.logger.debug("PS updated model")

            self.future_model = torch.futures.Future()
        return fut


class Worker(object):
    """
    This class performs the computation of the gradient and sends it back to the master
    """

    def __init__(self, ps_rref, dataloader, name, epochs, logger):
        self.ps_rref = ps_rref
        self.loss_fn = nn.functional.nll_loss
        self.dataloader = dataloader
        self.name = name
        self.epochs = epochs
        self.logger = logger
        logger.debug(f"Initialize trainer {name}")

    def get_next_batch(self):
        for i in range(self.epochs):
            for inputs, labels in tqdm(
                self.dataloader, position=self.name, leave=False, desc=f"Epoch {i}"
            ):
                yield inputs, labels, i

    def train(self):
        m = self.ps_rref.rpc_sync().get_model(self.name)
        for inputs, labels, i in self.get_next_batch():
            loss = self.loss_fn(m(inputs), labels)
            loss.backward()
            loss = loss.detach().sum()
            self.logger.debug(f"Epoch: {i}, Loss: {loss}")
            # Apply PCA to loss to get 2D
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer.update_and_fetch_model,
                args=(
                    self.ps_rref,
                    [p.grad for p in m.parameters()],
                    self.name,
                    loss,
                    i,
                ),
            )


def run_worker(ps_rref, dataloader, name, epochs, output_folder, logger):
    """
    Individually initialize the trainers with their loggers and start training
    """
    trainer = Worker(ps_rref, dataloader, name, epochs, logger)
    logger.debug(f"Run trainer {name}")
    trainer.train()


def run_parameter_server(
    trainers,
    output_folder,
    dataset,
    batch_size,
    epochs,
    delay,
    learning_rate,
    momentum,
    model_accuracy,
    save_model,
    mode,
    pca_gen,
    dataset_name, logger
):
    """
    This is the main function which launches the training of all the slaves
    """
    logger.info("Start training")

    ps_rref = rpc.RRef(
        ParameterServer(
            delay,
            len(trainers),
            learning_rate,
            momentum,
            logger,
            mode,
            pca_gen,
            output_folder,
            dataset_name
        )
    )
    futs = []

    # Random partition of the classes for each trainer
    classes = np.unique(dataset.targets)
    np.random.shuffle(classes)
    chunks = np.floor(len(classes) / len(trainers)).astype(int)
    targets = [classes[i * chunks : (i + 1) * chunks] for i in range(0, len(trainers))]

    for id, trainer in enumerate(trainers):
        logger.info(f"Trainer {id+1} working with classes {targets[id]}")

        # Dataset slicing for each worker
        d = copy.deepcopy(dataset)
        idx = np.isin(d.targets, targets[id])
        d.data = d.data[idx]
        d.targets = np.array(d.targets)[idx]
        dataloader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True)

        futs.append(
            rpc.rpc_async(
                trainer,
                run_worker,
                args=(
                    ps_rref,
                    dataloader,
                    id + 1,
                    epochs,
                    output_folder,
                    logger
                ),
            )
        )

    torch.futures.wait_all(futs)
    ps = ps_rref.to_here()

    np.savetxt(os.path.join(output_folder, "loss.txt"), ps.losses)
    np.savetxt(os.path.join(output_folder, "norms.txt"), ps.norms)

    logger.info("Finish training")

    if model_accuracy:
        train_loader_full = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size * 10, shuffle=True
        )
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
        filename = f"mnist_async_{learning_rate}_{momentum}_{batch_size}_{mode}_numclasses_{len(targets)*chunks}.pt"
        torch.save(
            ps_rref.to_here().model.state_dict(), os.path.join(output_folder, filename)
        )
        print(f"Model saved: {filename}")
    logging.shutdown()


def run(
    rank,
    world_size,
    output_folder,
    dataset,
    batch_size,
    epochs,
    delay,
    learning_rate,
    momentum,
    model_accuracy,
    save_model,
    mode,
    pca_gen,
    dataset_name
):
    """
    Creates the PS and launches the trainers
    """
    logger= setup_logger(log_queue)

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=6, rpc_timeout=0  # infinite timeout
    )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps", rank=rank, world_size=world_size, rpc_backend_options=options
        )
        run_parameter_server(
            [f"trainer{r}" for r in range(1, world_size)],
            output_folder,
            dataset,
            batch_size,
            epochs,
            delay,
            learning_rate,
            momentum,
            model_accuracy,
            save_model,
            mode,
            pca_gen,
            dataset_name,logger
        )

    # block until all rpcs finish
    rpc.shutdown()





if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Synchronous Parallel SGD parameter-Server RPC based training")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes [2,+inf].""")
    parser.add_argument(
        "--train_split",
        type=float,
        default=None,
        help="""Percentage of the training dataset to be used for training (0,1].""")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="""Learning rate of SGD  (0,+inf).""")
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="""Momentum of SGD  [0,+inf).""")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""")
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="""If set, the trained model will not be saved.""")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="""Number of epochs for training [1,+inf).""")
    parser.add_argument(
        "--model_accuracy",
        action="store_true",
        help="""If set, will compute the train accuracy of the global model after training.""")
    parser.add_argument(
        "--worker_accuracy",
        action="store_true",
        help="""If set, will compute the train accuracy of each worker after training """)
    parser.add_argument(
        "-pca_gen",
        help="Generate data for the PCA matrix generation",
        action="store_true",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        help="""Choose amongst {compensation, scheduler swa}, scheduler indicates to only use a learning rate scheduler""",
    )
    args = parser.parse_args()
    output_folder = args.output
    delay = args.delay
    save_model = not(args.no_save_model)
    model_accuracy = args.m 
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = args.momemtum
    dataset_name = args.dataset
    mode = args.mode
    pca_gen = args.pca_gen
    world_size = args.world_size
    train_split = args.train_split
    lr = args.lr
    momentum = args.momentum
    epochs = args.epochs
    model_accuracy = args.model_accuracy
    worker_accuracy = args.worker_accuracy



    if dataset_name == "mnist":
        print("Using MNIST dataset")
        dataset = torchvision.datasets.MNIST(
            "./../data/mnist_data",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # normalize inputs
                ]
            ),
        )


    elif dataset_name == "cifar10":
        print("Using CIFAR10 dataset")
        dataset = torchvision.datasets.CIFAR10(
            "./../data/cifar10",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # normalize inputs
                ]
            ),
        )

    elif dataset_name == "cifar100":
        print("Using CIFAR100 dataset")
        dataset = torchvision.datasets.CIFAR100(
            "./../data/cifar100",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # normalize inputs
                ]
            ),
        )

    elif dataset_name == "fmnist":
        print("Using FashionMNIST dataset")
        dataset = torchvision.datasets.FashionMNIST(
            "./../data/fmnist",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),  # first, convert image to PyTorch tensor
                    torchvision.transforms.Normalize(
                        (0.1307,), (0.3081,)
                    ),  # normalize inputs
                ]
            ),
        )


    else:
        print("You must specify a dataset amongst: {mnist, cifar10, cifar100, fmnist}")
        exit()

    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)

    if not (
        os.path.exists(
            os.path.join(
                output_folder,
                f"nn_async_{learning_rate}_{momentum}_{batch_size}_{mode}",
            )
        )
    ):
        os.mkdir(
            os.path.join(
                output_folder,
                f"nn_async_{learning_rate}_{momentum}_{batch_size}_{mode}",
            )
        )
    output_folder = os.path.join(
        output_folder, f"nn_async_{learning_rate}_{momentum}_{batch_size}_{mode}"
    )

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    if world_size is None:
        world_size = DEFAULT_WORLD_SIZE
        print(f"Using default world_size value: {DEFAULT_WORLD_SIZE}")
    elif world_size < 2:
        print("Forbidden value !!! world_size must be >= 2 (1 Parameter Server and 1 Worker)")
        exit()

    if train_split is None:
        train_split = DEFAULT_TRAIN_SPLIT
        print(f"Using default train_split value: {DEFAULT_TRAIN_SPLIT}")
    elif train_split > 1 or train_split <= 0:
        print("Forbidden value !!! train_split must be between (0,1]")
        exit()

    if lr is None:
        lr = DEFAULT_LR
        print(f"Using default lr: {DEFAULT_LR}")
    elif lr <= 0:
        print("Forbidden value !!! lr must be between (0,+inf)")
        exit()

    if momentum is None:
        momentum = DEFAULT_MOMENTUM
        print(f"Using default momentum: {DEFAULT_MOMENTUM}")
    elif momentum < 0:
        print("Forbidden value !!! momentum must be between [0,+inf)")
        exit()

    if epochs is None:
        epochs = DEFAULT_EPOCHS
        print(f"Using default epochs: {DEFAULT_EPOCHS}")
    elif epochs < 1:
        print("Forbidden value !!! epochs must be between [1,+inf)")
        exit()



    with Manager() as manager:
        log_queue = manager.Queue()
        log_writer_thread = threading.Thread(target=log_writer, args=(log_queue, output_folder,))

        log_writer_thread.start()
        mp.spawn(run, args=(
            world_size,
            output_folder,
            dataset,
            batch_size,
            epochs,
            delay,
            learning_rate,
            momentum,
            model_accuracy,
            save_model,
            mode,
            pca_gen,dataset_name,), nprocs=args.world_size, join=True)

        log_queue.put(None)
        log_writer_thread.join()


"""
Digit 0: 5923 batches
Digit 1: 6742 batches
Digit 2: 5958 batches
Digit 3: 6131 batches
Digit 4: 5842 batches
Digit 5: 5421 batches
Digit 6: 5918 batches
Digit 7: 6265 batches
Digit 8: 5851 batches
Digit 9: 5949 batches
"""
