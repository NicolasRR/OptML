import os
import threading
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
from torch import optim
import torchvision
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
DEFAULT_WORLD_SIZE = 4
BATCH_SIZE = 32
TRAIN_LOADER = #MNIST train
TEST_LOADER = #MNIST test
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #model layers
    def forward(self, x):
        #model forward
        return output
class BatchUpdateParameterServer(object):
    def __init__(self, batch_update_size):
        self.model = Net()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
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
        for p, g in zip(self.model.parameters(), grads):
            if (p.grad is not None )and (g is not None):
                p.grad += g 
        self.loss = np.append(self.loss, loss)
        with self.lock:
            self.curr_update_size += 1
            fut = self.future_model
            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad /= self.batch_update_size
                    else:
                        self.optimizer.zero_grad()
                self.curr_update_size = 0
                self.losses = np.append(self.losses, self.loss.mean())
                self.loss = np.array([])
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=False)
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()
        return fut
class Trainer(object):
    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.functional.nll_loss
    def get_next_batch(self):
        for (inputs,labels) in tqdm(TRAIN_LOADER):
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
def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()
def run_ps(trainers, batch_update_size):
    ps_rref = rpc.RRef(BatchUpdateParameterServer(batch_update_size))
    futs = []
    for trainer in trainers:
        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,))
        )
    torch.futures.wait_all(futs)
def run(rank, world_size):
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
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([f"trainer{r}" for r in range(1, world_size)], world_size-1)
    rpc.shutdown()
if __name__=="__main__":
    #main