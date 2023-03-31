import os
import threading
import queue
import torch
import torch.nn as nn
from torch import optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from multiprocessing import Manager
import argparse
from tqdm import tqdm
import logging
import logging.handlers
import numpy as np
import matplotlib.pyplot as plt #remove if unused
import random
import copy

DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_BATCH_SIZE = 32 # 1 == SGD, >1 MINI BATCH SGD
DEFAULT_EPOCHS = 1

#################################### LOGGER ####################################
def setup_logger(log_queue):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    qh = QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    #fh = logging.FileHandler("log.log")
    #fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    qh.setFormatter(formatter)
    #fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(qh)
    #logger.addHandler(fh)

    return logger

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

def log_writer(log_queue):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    with open('log.log', 'w') as log_file:
        while True:
            try:
                record = log_queue.get(timeout=1) 
                if record is None:
                    break
                msg = formatter.format(record)
                log_file.write(msg + "\n")
            except queue.Empty:
                continue
    
#################################### NET ####################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

#################################### PARAMETER SERVER ####################################
class ParameterServer(object):

    def __init__(self, batch_update_size, logger, learning_rate, momentum):
        self.model = Net()
        self.logger = logger
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum= momentum)
        self.losses = np.array([])
        self.loss = np.array([])
        for params in self.model.parameters():
            params.grad = torch.zeros_like(params)
        
    def get_model(self):
        return self.model
    
    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, worker_name, worker_batch_count, worker_epoch, total_batches_to_run, total_epochs, loss):
        self = ps_rref.local_value()
        self.logger.debug(f"PS got {self.curr_update_size +1}/{self.batch_update_size} updates (from {worker_name}, {worker_batch_count}/{total_batches_to_run}, epoch {worker_epoch}/{total_epochs})")
        for param, grad in zip(self.model.parameters(), grads):
            if (param.grad is not None )and (grad is not None):
                param.grad += grad
            elif(param.grad is None):
                self.logger.debug(f"None param.grad detected from worker {worker_name}")
            else: 
                self.logger.debug("None grad detected")
        self.loss = np.append(self.loss, loss)
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad /= self.batch_update_size
                    else:
                        self.logger.debug(f"None param.grad detected for the update")
                        self.optimizer.zero_grad()
                self.curr_update_size = 0
                self.losses = np.append(self.losses, self.loss.mean())
                self.logger.debug(f"Global model loss is {self.losses[-1]}")
                self.loss = np.array([])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)
                fut.set_result(self.model)
                self.logger.debug("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


#################################### WORKER ####################################
class Worker(object):

    def __init__(self, ps_rref, train_loader, logger, epochs, worker_accuracy):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_fn = nn.functional.nll_loss
        self.logger = logger
        self.batch_count = 0
        self.current_epoch = 0
        self.epochs = epochs
        self.worker_name = rpc.get_worker_info().name
        self.worker_accuracy = worker_accuracy
        self.logger.debug(f"{self.worker_name} is working on a dataset of size {len(train_loader.sampler)}") #length of the subtrain set
        #self.logger.debug(f"{self.worker_name} is working on a dataset of size {len(train_loader)}") #total number of batches to run (len subtrain set / batch size)

    def get_next_batch(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch +1
            if self.worker_name == "Worker_1":
                iterable = tqdm(self.train_loader)
            else:
                iterable = self.train_loader

            for (inputs, labels) in iterable:
                yield inputs, labels

    def train(self):
        worker_model = self.ps_rref.rpc_sync().get_model()

        for inputs, labels in self.get_next_batch():
            #print(labels, self.worker_name) #check the samples of workers
            loss = self.loss_fn(worker_model(inputs), labels)
            loss.backward()
            self.batch_count += 1

            worker_model = rpc.rpc_sync(
                self.ps_rref.owner(),
                ParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [param.grad for param in worker_model.parameters()], self.worker_name, self.batch_count, self.current_epoch, len(self.train_loader), self.epochs, loss.detach().sum()),
            )
            if self.worker_accuracy:
                if self.batch_count == len(self.train_loader) and self.current_epoch == self.epochs:
                    correct_predictions = 0
                    total_predictions = 0
                    with torch.no_grad():  # No need to track gradients for evaluation
                        for _, (data, target) in enumerate(self.train_loader):
                            logits = worker_model(data)
                            predicted_classes = torch.argmax(logits, dim=1)
                            correct_predictions += (predicted_classes == target).sum().item()
                            total_predictions += target.size(0)
                        final_train_accuracy = correct_predictions / total_predictions
                    print(f"Accuracy of {self.worker_name}: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})")


#################################### GLOBAL FUNCTIONS ####################################
def run_worker(ps_rref, train_loader, logger, epochs, worker_accuracy):
    worker = Worker(ps_rref, train_loader, logger, epochs, worker_accuracy)
    worker.train()


def run_parameter_server(workers, batch_update_size, unique_datasets, logger, learning_rate, momentum, save_model=True, train_split=DEFAULT_TRAIN_SPLIT, batch_size=None, epochs=DEFAULT_EPOCHS, worker_accuracy=False, model_accuracy=False, digits=None):

    train_data = torchvision.datasets.MNIST('data/', 
                                        download=True, 
                                        train=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    # Shuffle and split the train_data
    train_data_indices = torch.randperm(len(train_data))
    train_length = int(train_split * len(train_data))
    subsample_train_indices = train_data_indices[:train_length]
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif batch_size < 1 or batch_size > train_length:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    
    train_loader_full = 0 #train loader for final global accuracy, useful when workers don't share samples
    ps_rref = rpc.RRef(ParameterServer(batch_update_size, logger, learning_rate, momentum))
    futs = []

    if not unique_datasets and digits is None: #workers sharing samples
        train_loader  = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_train_indices)) 
        if model_accuracy:
            train_loader_full= train_loader
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            futs.append(
                rpc.rpc_async(worker, run_worker, args=(ps_rref, train_loader, logger, epochs, worker_accuracy))
            )
    elif unique_datasets and digits is None:
        worker_indices = subsample_train_indices.chunk(batch_update_size) # Split the train_indices based on the number of workers (world_size - 1)
        if model_accuracy:
            train_loader_full= DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_train_indices)) 
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            train_loader  = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(worker_indices[idx])) 
            futs.append(
                rpc.rpc_async(worker, run_worker, args=(ps_rref, train_loader, logger, epochs, worker_accuracy))
            )
    elif digits is not None:
        random_digits = random.sample(range(10), digits) #randomly creating could be replaced with terminal input
        print(f"Each worker will be assigned to one of the following digits: {random_digits}")
        digits_indices = []
        for _ in random_digits:
            digits_indices.append([])
        for i in range(len(subsample_train_indices)):
            _, label = train_data[subsample_train_indices[i]]
            for j, digit in enumerate(random_digits):
                if digit == label:
                    digits_indices[j].append(subsample_train_indices[i])
        len_min_subset = train_length
        for i in range(len(digits_indices)):
            if len(digits_indices[i]) < len_min_subset:
                len_min_subset = len(digits_indices[i])
        train_loaders_digits = []
        for i in range(len(digits_indices)):
            digit_indices = copy.deepcopy(digits_indices[i])
            random.shuffle(digit_indices) 
            digit_indices = digit_indices[:len_min_subset] #sync mode, datasets must be same length (find the smallest dataset and slice the other ones)
            digits_indices[i] = digit_indices
            train_loaders_digits.append(DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(digit_indices)))
        if model_accuracy:
            full_digits_list = []
            for sublist in digits_indices:
                full_digits_list.extend(sublist)
            train_loader_full= DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(full_digits_list)) 
        logger.info("Start training")
        for idx, worker in enumerate(workers):
            train_loader = train_loaders_digits[idx]
            futs.append(
                rpc.rpc_async(worker, run_worker, args=(ps_rref, train_loader, logger, epochs, worker_accuracy))
            )

    torch.futures.wait_all(futs)

    losses = ps_rref.to_here().losses
    logger.info("Finished training")
    print(f"Final train loss: {losses[-1]}")
    #plt.plot(range(len(losses)), losses)
    #plt.xlabel("Losses")
    #plt.ylabel("Update steps")
    #plt.savefig("loss.png")

    if model_accuracy:
        correct_predictions = 0
        total_predictions = 0
        #memory efficient way (for large datasets)
        with torch.no_grad():  # No need to track gradients for evaluation
            for _, (data, target) in enumerate(train_loader_full):
                logits = ps_rref.to_here().model(data)
                predicted_classes = torch.argmax(logits, dim=1)
                correct_predictions += (predicted_classes == target).sum().item()
                total_predictions += target.size(0)
        final_train_accuracy = correct_predictions / total_predictions
        print(f"Final train accuracy: {final_train_accuracy*100} % ({correct_predictions}/{total_predictions})")

    if save_model:
        if unique_datasets:
            filename = f"mnist_sync_{batch_update_size+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_split_datasets.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")
        elif  digits is not None:
            filename = f"mnist_sync_{batch_update_size+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_digits_{''.join(str(i) for i in random_digits)}.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")
        else:
            filename = f"mnist_sync_{batch_update_size+1}_{str(train_split).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}.pt"
            torch.save(ps_rref.to_here().model.state_dict(), filename)
            print(f"Model saved: {filename}")



def run(rank, world_size, learning_rate, momentum, log_queue, save_model, unique_datasets, train_split, batch_size, epochs, worker_accuracy, model_accuracy, digits):
    logger= setup_logger(log_queue)
    rpc_backend_options= rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=world_size,
        rpc_timeout=0 # infinite timeout
    )

    if rank != 0:
        #starting up worker
        rpc.init_rpc(
            f"Worker_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        # worker passively waiting for parameter server to kick off training iterations
    else:
        # parameter server gives data to the workers
        rpc.init_rpc(
            "Parameter_Server",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options
        )
        run_parameter_server([f"Worker_{r}" for r in range(1, world_size)], world_size-1, unique_datasets, logger, learning_rate, momentum, save_model, train_split, batch_size, epochs, worker_accuracy, model_accuracy, digits)

    # block until all rpcs finish
    rpc.shutdown()


#################################### MAIN ####################################
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
        default="0.0.0.0",
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
        "--unique_datasets",
        action="store_true",
        help="""After applying train_split, each worker will train on a unique distinct dataset (samples will not be 
        shared between workers).""")
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
        help="""If set, will compute the train accuracy of each worker after training (useful when --unique_datasets).""")
    parser.add_argument(
        "--digits",
        type=int,
        default=None,
        help="""Reprensents the amount of digits that will be trained in parallel, it will split the MNIST dataset in {digits} parts, one part per digit, and each part will be assigned to a worker.
        This mode requires --world_size {digits +1} --batch_size 1, don't use --unique_datasets.""")

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    if args.world_size is None:
        args.world_size = DEFAULT_WORLD_SIZE
        print(f"Using default world_size value: {DEFAULT_WORLD_SIZE}")
    elif args.world_size < 2:
        print("Forbidden value !!! world_size must be >= 2 (1 Parameter Server and 1 Worker)")
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

    if args.unique_datasets:
        unique_datasets = True
    else:
        unique_datasets = False

    if args.model_accuracy:
        model_accuracy = True
    else:
        model_accuracy = False

    if args.worker_accuracy:
        worker_accuracy = True
    else:
        worker_accuracy = False

    if args.digits is not None:
        if args.digits < 1 or args.digits > 10:
            print("Forbidden value !!! digits must be between [1,10]")
            exit()
        elif unique_datasets:
            print("Please use --digits without --unique_datasets")
            exit()
        elif args.world_size != args.digits+1:
            print("Please use --digits with --world_size {digits+1}")
            exit()
        elif args.batch_size != 1:
            print("Please use --digits with the --batch_size 1")
            exit()


    with Manager() as manager:
        log_queue = manager.Queue()
        log_writer_thread = threading.Thread(target=log_writer, args=(log_queue,))

        log_writer_thread.start()
        mp.spawn(run, args=(args.world_size, args.lr, args.momentum, log_queue, save_model, unique_datasets, args.train_split, args.batch_size, args.epochs, worker_accuracy, model_accuracy, args.digits), nprocs=args.world_size, join=True)

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