import os
import threading
from datetime import datetime
import argparse


import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms


import torchvision


BATCH_SIZE = 20
image_w = 64
image_h = 64
num_classes = 30
BATCH_UPDATE_SIZE = 5
num_batches = 6

TRAIN_LOADER = torch.utils.data.DataLoader(datasets.MNIST('../data/mnist_data', 
                                                    download=True, 
                                                    train=True,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                    ])), 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True)
TEST_LOADER = torch.utils.data.DataLoader(datasets.MNIST('../data/mnist_data', 
                                                    download=True, 
                                                    train=False,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                    ])), 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True)


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer(object):

    def __init__(self, BATCH_UPDATE_SIZE=BATCH_UPDATE_SIZE):
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.BATCH_UPDATE_SIZE = BATCH_UPDATE_SIZE
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
        timed_log(f"PS got {self.curr_update_size}/{BATCH_UPDATE_SIZE} updates")
        for p, g in zip(self.model.parameters(), grads):
            p.grad += g
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.BATCH_UPDATE_SIZE:
                for p in self.model.parameters():
                    p.grad /= self.BATCH_UPDATE_SIZE
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer(object):

    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(BATCH_SIZE) \
                                    .random_(0, num_classes) \
                                    .view(BATCH_SIZE, 1)

    def get_next_batch(self):
        for _ in range(num_batches):
            inputs = torch.randn(BATCH_SIZE, 3, image_w, image_h)
            labels = torch.zeros(BATCH_SIZE, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
            yield inputs.cuda(), labels.cuda()

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model().cuda()
        for inputs, labels in self.get_next_batch():
            timed_log(f"{name} processing one batch")
            self.loss_fn(m(inputs), labels).backward()
            timed_log(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).cuda()
            timed_log(f"{name} got updated model")


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    ps_rref = rpc.RRef(BatchUpdateParameterServer())
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )

    torch.futures.wait_all(futs)
    timed_log("Finish training")


def run(rank, world_size):

    options=rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # infinite timeout
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
        default=4,
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

    # world_size = args.world_size
    # BATCH_UPDATE_SIZE = world_size-1
    world_size = BATCH_UPDATE_SIZE+1
    num_batches = world_size
    mp.spawn(run, args=(world_size, ), nprocs=world_size, join=True)
