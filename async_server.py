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

logging.basicConfig(level=logging.INFO)



BATCH_SIZE = 32
image_w = 64
image_h = 64
num_classes = 10
BATCH_UPDATE_SIZE = 5
NUM_BATCHES = 6
DEVICE = "cpu"

def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")

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

        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(DEVICE)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(DEVICE)
        self.dropout1 = nn.Dropout2d(0.25).to(DEVICE)
        self.dropout2 = nn.Dropout2d(0.5).to(DEVICE)
        self.fc1 = nn.Linear(9216, 128).to(DEVICE)
        self.fc2 = nn.Linear(128, 10).to(DEVICE)

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

    def __init__(self, batch_update_size=BATCH_UPDATE_SIZE):
        self.model = Net()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        logging.debug(f"PS got {self.curr_update_size}/{BATCH_UPDATE_SIZE} updates")
        for p, g in zip(self.model.parameters(), grads):
            #print("grads :", grads)
            if (p.grad is not None )and (g is not None):
                p.grad += g
            elif(p.grad is None):
                logging.warning("None p.grad detected")
            else: 
                logging.warning("None g detected")
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                logging.debug("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.functional.nll_loss


    def get_next_batch(self):
        for (inputs,labels) in tqdm(TRAIN_LOADER):

            yield inputs.to(DEVICE), labels.to(DEVICE)

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model().to(DEVICE)
        for inputs, labels in self.get_next_batch():
            logging.debug(f"{name} processing one batch")
            # print("Shape of network output ", m(inputs).shape)
            # print("Shape of labels ", labels.shape)
            self.loss_fn(m(inputs), labels).backward()
            logging.debug(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).to(DEVICE)
            logging.debug(f"{name} got updated model")


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    logging.info("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    logging.info("Finish training")


def run(rank, world_size):

    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=6,
        rpc_timeout=5  # infinite timeout
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
        default="localhost",
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
    BATCH_UPDATE_SIZE = world_size-1
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
