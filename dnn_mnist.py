import os
import threading
from datetime import datetime

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


TRAIN_LOADER = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./../data/mnist_data', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ])), 
                                batch_size=BATCH_SIZE, 
                                shuffle=True)
TEST_LOADER = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./../data/mnist_data', 
                                                    download=True, 
                                                    train=False,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                        torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                    ])), 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True)

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
        self.dataloader = TRAIN_LOADER
        logger.info(f"number of layers is {len(self.lnorms)}")

    def get_model(self):
        return self.model
    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads,id, loss):
        self = ps_rref.local_value()
        logger.debug(f"PS got 1 updates from worker {id}")
        self.trainers[int(id[-1])-1] +=1
        logger.debug(f"The value of trainer is {self.trainers}")
        self.losses = np.append(self.losses, loss)
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

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.functional.nll_loss


    def get_next_batch(self, loader):
        for (inputs,labels) in tqdm(loader):

            yield inputs, labels

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model()
        loader = self.ps_rref.rpc_sync().get_dataloader()
        for (inputs, labels) in self.get_next_batch(loader):
            loss = self.loss_fn(m(inputs), labels)
            loss.backward()
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.parameters()], name, loss.detach().sum()),
            )


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    logger.info("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    ps = ps_rref.to_here()
    losses = ps.losses
    norms = ps.norms
    lnorms = ps.lnorms
    trainers = ps.trainers
    np.savetxt("trainers.txt", trainers)
    n = len(losses)
    plt.plot(range(n), losses)
    plt.xlabel("Losses")
    plt.ylabel("Update steps")
    plt.savefig("loss.png")
    np.savetxt("loss.txt", losses)
    plt.figure()
    plt.plot(range(n), norms)
    plt.ylabel("Norms")
    plt.xlabel("Update steps")
    plt.savefig("norms.png")
    np.savetxt("norms.txt", norms)
    plt.figure()
    for i in range(len(lnorms)):
        l = lnorms[i]
        plt.plot(range(n), l, label=f"Layer {i}")
        np.savetxt(f"lnorms{i}.txt", l)
    plt.ylabel("Norms")
    plt.xlabel("Update steps")
    plt.savefig("lnorms.png")
    logger.info("Finish training")


def run(rank, world_size):

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
