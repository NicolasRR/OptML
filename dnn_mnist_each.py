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
import numpy as np
import matplotlib.pyplot as plt
import copy

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

## TO DO: implement standarization

BATCH_SIZE = 32

NUM_BATCHES = 6
DEVICE = "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
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
    """
    This class contains the parameters and updates them at every iteration. To do so it performs thread locking in order to avoid memory issues. 

    """

    def __init__(self):
        self.model = Net()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        self.losses = np.array([])
        self.norms = np.array([])
        for i,p in enumerate(self.model.parameters()):
            p.grad = torch.zeros_like(p)
        self.lnorms = [[] for j in range(i+1)]
        self.trainers = np.zeros(5, dtype=int)        

    def get_model(self):
        return self.model


    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads,id, loss):
        """
        This function can be called asynchonously by all the slaves in order to update the master's parameters. It manually add the gradient to .grad
        """
        self = ps_rref.local_value()
        logger.debug(f"PS got 1 updates from worker {id}")
        self.trainers[id-1] +=1
        logger.debug(f"The value of trainer is {self.trainers}")
        self.losses = np.append(self.losses, loss)
        if id == 1:
            time.sleep(0.05)
        with self.lock:
            fut = self.future_model
            norm = 0
            for i,(p, g) in enumerate(zip(self.model.parameters(), grads)):
                if p.grad is not None:
                    p.grad += g
                    n = p.grad.norm(2).item() ** 2
                    norm += n
                    self.lnorms[i].append(n)
                else:
                    logger.debug(f"None p.grad detected for the update")
            logger.debug(f"Loss is {self.losses[-1]}")
            self.norms = np.append(self.norms, norm**(0.5))
            logger.debug(f"The norm of the gradient is {self.norms[-1]}")
            self.loss = np.array([])
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=False)
            fut.set_result(self.model)
            logger.debug("PS updated model")
            self.future_model = torch.futures.Future()

        return fut


class Trainer(object):
    """
    This class performs the computation of the gradient and sends it back to the master
    """

    def __init__(self, ps_rref, dataloader, name):
        self.ps_rref = ps_rref
        self.loss_fn = nn.functional.nll_loss
        self.dataloader = dataloader
        self.name = name


    def get_next_batch(self, p):
        for (inputs,labels) in tqdm(self.dataloader, position=p):
            yield inputs, labels

    def train(self):
        m = self.ps_rref.rpc_sync().get_model()
        for inputs, labels in self.get_next_batch(self.name):
            loss = self.loss_fn(m(inputs), labels)
            loss.backward()
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.parameters()], self.name, loss.detach().sum()),
            )


def run_trainer(ps_rref, dataloader, name):
    """
    Individually initialize the trainers and start training
    """
    trainer = Trainer(ps_rref, dataloader, name)
    trainer.train()


def run_ps(trainers):
    """
    This is the main function which launches the training of all the slaves
    """
    logger.info("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    dataset = torchvision.datasets.MNIST('./../data/mnist_data', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
    for id, trainer in enumerate(trainers):
        d = copy.deepcopy(dataset)
        idx = d.targets==id
        d.data = d.data[idx]
        d.targets = d.targets[idx]
        dataloader = torch.utils.data.DataLoader(d, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)

        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,dataloader, id+1,))
        )

    torch.futures.wait_all(futs)
    ps = ps_rref.to_here()
    np.savetxt("loss.txt", ps.losses)
    lnorms = ps.lnorms
    np.savetxt("norms.txt", ps.norms)
    print("lnorms size", len(lnorms))
    for i in range(len(lnorms)):
        np.savetxt(f"lnorms{i}.txt", lnorms[i])
    logger.info("Finish training")


def run(rank, world_size):
    """
    Creates the PS and launches the trainers
    """

    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=6,
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
        run_ps([f"trainer{r}" for r in range(1, world_size)])

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Asynchronous-parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=6,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
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


    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = args.world_size
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
