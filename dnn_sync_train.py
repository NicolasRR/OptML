import os
import threading
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from multiprocessing import Manager
import argparse
from tqdm import tqdm
import numpy as np
from helpers import create_worker_trainloaders, log_writer, setup_logger, _get_model, get_optimizer, get_scheduler, get_model_accuracy, _save_model, save_weights, compute_weights_l2_norm
from helpers import DEFAULT_DATASET, DEFAULT_WORLD_SIZE, DEFAULT_TRAIN_SPLIT, DEFAULT_LR, DEFAULT_MOMENTUM, DEFAULT_EPOCHS, DEFAULT_SEED, LOSS_FUNC


#################################### PARAMETER SERVER ####################################
class ParameterServer(object):
    def __init__(self, nb_workers, logger, dataset_name, learning_rate, momentum, use_alr, len_trainloader, epochs, lrs, saves_per_epoch):
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
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)
        print(f"Save idx: {save_idx}")

    def get_model(self):
        return self.model  # get global model
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']

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
            f"PS got {self.update_counter +1}/{self.nb_workers} updates (from {worker_name}, {worker_batch_count - total_batches_to_run*(worker_epoch-1)}/{total_batches_to_run} ({worker_batch_count}/{total_batches_to_run*total_epochs}), epoch {worker_epoch}/{total_epochs})"
        )
        for param, grad in zip(self.model.parameters(), grads):
            if (param.grad is not None) and (
                grad is not None
            ):  # remove if confident, good for security
                param.grad += grad  # accumulate workers grads

            elif param.grad is None:  # PyTorch security
                self.logger.debug(f"None param.grad detected from worker {worker_name}")
            else:
                self.logger.debug("None grad detected")

        self.loss = np.append(self.loss, loss)

        with self.lock:
            self.update_counter += 1
            fut = self.future_model

            if (
                self.update_counter >= self.nb_workers
            ):  # received grads from all workers
                for param in self.model.parameters():
                    if param.grad is not None:  # remove if confident, good for security
                        param.grad /= self.nb_workers  # average workers grads
                    else:  # remove if confident, good for security
                        self.logger.debug(f"None param.grad detected for the update")
                        self.optimizer.zero_grad()  # redundant
                self.update_counter = 0
                self.model_loss = self.loss.mean()  # aggregate the workers loss
                self.loss = np.array([])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)  # reset grad tensor to 0
                if self.saves_per_epoch is not None:
                    relative_batch_idx = worker_batch_count - total_batches_to_run*(worker_epoch-1) # PROBLEM HERE
                    if relative_batch_idx in self.save_idx:
                        weights = [w.detach().clone().cpu().numpy() for w in self.model.parameters()]
                        self.weights_matrix.append(weights)
                if worker_batch_count == total_batches_to_run:
                    if self.scheduler is not None:
                        self.scheduler.step()
                fut.set_result(self.model)
                self.logger.debug(f"PS updated model, global loss is {self.model_loss}, weights norm is {compute_weights_l2_norm(self.model)}")
                self.future_model = torch.futures.Future()

        return fut


#################################### WORKER ####################################
class Worker(object):
    def __init__(self, ps_rref, logger, train_loader, epochs, worker_accuracy):
        self.ps_rref = ps_rref
        self.train_loader = train_loader  # worker trainloader
        self.loss_func = LOSS_FUNC  # worker loss
        self.logger = logger
        self.batch_count = 0
        self.current_epoch = 0
        self.epochs = epochs
        self.worker_name = rpc.get_worker_info().name
        self.worker_accuracy = worker_accuracy
        self.logger.debug(
            f"{self.worker_name} is working on a dataset of size {len(train_loader.sampler)}"
        )

    def get_next_batch(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            if self.worker_name == "Worker_1":
                # progress bar only of the first worker (we are in synchronous mode)
                iterable = tqdm(
                    self.train_loader,
                    unit="batch",
                )
                current_lr = self.ps_rref.rpc_sync().get_current_lr()
                iterable.set_postfix(epoch=f"{self.current_epoch}/{self.epochs}", lr=f"{current_lr:.5f}")
            else:
                iterable = self.train_loader

            for inputs, labels in iterable:
                yield inputs, labels

        if self.worker_name == "Worker_1":
            iterable.close()

    def train(self):
        worker_model = self.ps_rref.rpc_sync().get_model()

        for inputs, labels in self.get_next_batch():
            # print(labels, self.worker_name) #check the samples of workers
            loss = self.loss_func(worker_model(inputs), labels)  # worker loss
            loss.backward()
            self.batch_count += 1

            worker_model = rpc.rpc_sync(
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
                    loss.detach().sum(),
                ),
            )
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


def run_parameter_server(
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
    saves_per_epoch, # to use, to worker or Parameter server class
    lrs,
):
    train_loaders, batch_size = create_worker_trainloaders(
        len(workers),
        dataset_name,
        split_dataset,
        split_labels,
        False,  # compatibility for async split_labels_unscaled
        train_split,
        batch_size,
        model_accuracy,
    )
    train_loader_full = None
    if model_accuracy:
        train_loader_full = train_loaders[1]
        train_loaders = train_loaders[0]

    ps_rref = rpc.RRef(
        ParameterServer(len(workers), logger, dataset_name, learning_rate, momentum, use_alr, len(train_loaders), epochs, lrs, saves_per_epoch)
    )
    futs = []

    if not split_dataset and not split_labels:  # workers sharing samples
        logger.info(f"Starting synchronous SGD training with {len(workers)} workers")
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
    print(f"Final train loss: {ps_rref.to_here().model_loss}")

    if model_accuracy:
        get_model_accuracy(ps_rref.to_here().model, train_loader_full)

    if save_model:
        _save_model("sync", dataset_name, ps_rref.to_here().model, len(workers), train_split, learning_rate, momentum, batch_size, epochs, subfolder, split_dataset, split_labels)

    if saves_per_epoch is not None:
        save_weights(ps_rref.to_here().weights_matrix, "sync", dataset_name, train_split, learning_rate, momentum, batch_size, epochs, subfolder)


def run(
    rank,
    log_queue,
    dataset_name,
    split_dataset,
    split_labels,
    world_size,
    learning_rate,
    momentum,
    train_split,
    batch_size,
    epochs,
    worker_accuracy,
    model_accuracy,
    save_model,
    subfolder,
    saves_per_epoch,
    use_alr,
    lrs,
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
        )

    # block until all rpcs finish
    rpc.shutdown()


#################################### MAIN ####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synchronous Parallel SGD parameter-Server RPC based training"
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
        default=None,
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
        Workers will not share samples and the labels are randomly assigned.
        This mode requires --batch_size 1, don't use --split_dataset. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
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
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="""Subfolder where the model and log_sync.log will be saved.""",
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
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if args.dataset is None:
        args.dataset = DEFAULT_DATASET
        print(f"Using default dataset: {DEFAULT_DATASET}")

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

    if args.split_labels:
        if args.split_dataset:
            print("Please use --split_labels without --split_dataset")
            exit()
        elif args.batch_size != 1:
            print("Please use --split_labels with the --batch_size 1")
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
        print(f"Saving model and log_sync.log to {args.subfolder}")

    if args.alr:
        print("Using Adam as optimizer instead of SGD")

    if args.lrs is not None:
        print(f"Using learning rate scheduler: {args.lrs}")

    with Manager() as manager:
        log_queue = manager.Queue()
        log_writer_thread = threading.Thread(
            target=log_writer, args=(log_queue, args.subfolder, "log_sync.log")
        )

        log_writer_thread.start()
        mp.spawn(
            run,
            args=(
                log_queue,
                args.dataset,
                args.split_dataset,
                args.split_labels,
                args.world_size,
                args.lr,
                args.momentum,
                args.train_split,
                args.batch_size,
                args.epochs,
                args.worker_accuracy,
                args.model_accuracy,
                not args.no_save_model,
                args.subfolder,
                args.saves_per_epoch,
                args.alr,
                args.lrs,
            ),
            nprocs=args.world_size,
            join=True,
        )

        log_queue.put(None)
        log_writer_thread.join()
