import os
import threading
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision

import argparse
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_WORLD_SIZE = 4
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.0
DEFAULT_BATCH_SIZE = 32

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('log.log', mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
    

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
        x = nn.functional.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

class BatchUpdateParameterServer(object):

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
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        
    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads,id, loss):
        self = ps_rref.local_value()
        self.logger.debug(f"PS got {self.curr_update_size +1}/{self.batch_update_size} updates")
        for p, g in zip(self.model.parameters(), grads):
            if (p.grad is not None )and (g is not None):
                p.grad += g
            elif(p.grad is None):
                self.logger.debug(f"None p.grad detected from worker {id}")
            else: 
                self.logger.debug("None g detected")
        self.loss = np.append(self.loss, loss)
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad /= self.batch_update_size
                    else:
                        self.logger.debug(f"None p.grad detected for the update")
                        self.optimizer.zero_grad()
                self.curr_update_size = 0
                self.losses = np.append(self.losses, self.loss.mean())
                self.logger.debug(f"Loss is {self.losses[-1]}")
                self.loss = np.array([])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)
                fut.set_result(self.model)
                self.logger.debug("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer(object):

    def __init__(self, ps_rref, train_loader, logger):
        self.ps_rref = ps_rref
        self.train_loader = train_loader
        self.loss_fn = nn.functional.nll_loss
        self.logger = logger
        worker_name = rpc.get_worker_info().name
        logger.info(f"Worker {worker_name} is working on a dataset of size {len(train_loader.sampler)}")
        #self.logger.debug(f"Worker {worker_name} is working on a dataset of size {len(train_loader.sampler)}")

    def get_next_batch(self):
        for (inputs,labels) in tqdm(self.train_loader):

            yield inputs, labels

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model()
        for inputs, labels in self.get_next_batch():
            loss = self.loss_fn(m(inputs), labels)
            loss.backward()
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.parameters()], name, loss.detach().sum()),
            )
        # Saving the model
        torch.save(m.state_dict(), 'mnist_sync.pt')


def run_trainer(ps_rref, train_loader, logger):
    trainer = Trainer(ps_rref, train_loader, logger)
    trainer.train()


def run_ps(trainers, batch_update_size, train_loader, logger, learning_rate, momentum):
    logger.info("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer(batch_update_size, logger, learning_rate, momentum))
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref, train_loader, logger))
        )

    torch.futures.wait_all(futs)
    losses = ps_rref.to_here().losses
    plt.plot(range(len(losses)), losses)
    plt.xlabel("Losses")
    plt.ylabel("Update steps")
    plt.savefig("loss.png")
    logger.info("Finished training")
    print(f"Final train loss: {losses[-1]}")



def run(rank, world_size, train_loader, learning_rate, momentum):
    logger = setup_logger()
    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=world_size,
        rpc_timeout=0 # infinite timeout
     )
    if rank != 0:
        rpc.init_rpc(
            f"trainer{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # trainer passively waiting for ps to kick off training iterations
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)], world_size-1, train_loader, logger, learning_rate, momentum)

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Asynchronous-parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--train_split",
        type=float,
        default=None,
        help="""Percentage of the training dataset to be used for training.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="0.0.0.0",
        help="""Address of master, will default to localhost if not provided.
        Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="29500",
        help="""Port that master is listening on, will default to 29500 if not
        provided. Master must be able to accept network traffic on the host and port.""")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="""Learning rate of SGD.""")
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="""Momentum of SGD.""")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of SGD.""")


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
    
    train_data = torchvision.datasets.MNIST('data/', 
                                        download=True, 
                                        train=True,
                                        transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    # Shuffle and split the train_data
    train_data_indices = torch.randperm(len(train_data))
    train_length = int(args.train_split * len(train_data))
    subsample_train_indices = train_data_indices[:train_length]
    train_loader  = DataLoader(train_data, batch_size=args.batch_size, sampler=SubsetRandomSampler(subsample_train_indices)) 

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1 or args.batch_size > train_length:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    mp.spawn(run, args=(args.world_size, train_loader, args.lr, args.momentum), nprocs=args.world_size, join=True)
