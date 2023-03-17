import argparse
import os
from threading import Lock
import time
import logging
import sys

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributed.optim import DistributedOptimizer
import torch.distributed as dist
from torchvision import datasets, transforms

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

torch.autograd.set_detect_anomaly(True)

# Constants
TRAINER_LOG_INTERVAL = 5  # How frequently to log information
TERMINATE_AT_ITER = 100  # for early stopping when debugging
PS_AVERAGE_EVERY_N = 25  # How often to average models between trainers


NUM_TRAINERS_WAITED = 0
num_trainers_waited_lock = Lock()

def trainer_arrived():
    with num_trainers_waited_lock:
        global NUM_TRAINERS_WAITED
        NUM_TRAINERS_WAITED += 1

# from threading import Condition
# trainer_cv = Condition()
# def set_cv():
#     global trainer_cv
#     with trainer_cv:
#         trainer_cv.notify()

def wait_all_trainers(rank, world_size):
    return
    global NUM_TRAINERS_WAITED
    # Send RPC to all other trainers (non-zero ranks)
    for i in range(world_size):
        if i != rank and i != 0:
            rpc.rpc_sync(f"trainer_{i}", trainer_arrived, args=())
    # Wait for all trainers to arrive
    with num_trainers_waited_lock:
        cur_num_trainers = NUM_TRAINERS_WAITED

    if cur_num_trainers != world_size - 1:
        time.sleep(0.01)
        with num_trainers_waited_lock:
            cur_num_trainers = NUM_TRAINERS_WAITED

    time.sleep(0.01)

    # Notify other trainers to unblock
    # rpc.rpc_sync("trainer_1", set_cv, args=())
    # global trainer_cv
    # with trainer_cv:
    #     trainer_cv.wait()


# --------- Helper Methods --------------------	

# On the local node, call a method with first arg as the value held by the	
# RRef. Other args are passed in as arguments to the function called.	
# Useful for calling instance methods.	
def call_method(method, rref, *args, **kwargs):	
    return method(rref.local_value(), *args, **kwargs)	

# Given an RRef, return the result of calling the passed in method on the value	
# held by the RRef. This call is done on the remote node that owns	
# the RRef. args and kwargs are passed into the method.	
# Example: If the value held by the RRef is of type Foo, then	
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling	
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result	
# back.	


def remote_method(method, rref, *args, **kwargs):	
    args = [method, rref] + list(args)	
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

# --------- MNIST Network to train, from pytorch/examples -----


class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        logger.info(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        logger.info(f"Putting first 2 convs on {str(device)}")
        # Put conv layers on the first cuda device
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        # Put rest of the network on the 2nd cuda device, if there is one
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")

        logger.info(f"Putting rest of layers on {str(device)}")
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # x = 9216x128
        # x = torch.randn((x.shape[0],  9216))
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.model = Net(num_gpus=num_gpus)
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def forward(self, rank, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes parameters remotely.
    def get_param_rrefs(self, rank):
        param_rrefs = [rpc.RRef(param)
                       for param in self.model.parameters()]
        return param_rrefs

    # def create_model_for_rank(self, rank, num_gpus):
    #     assert num_gpus == self.num_gpus, f"Inconsistent no. of GPUs requested from rank vs initialized with on PS: {num_gpus} vs {self.num_gpus}"
    #     with self.models_init_lock:
    #         if rank not in self.models:
    #             self.models[0] = Net(num_gpus=num_gpus)

    # def get_num_models(self):
    #     with self.models_init_lock:
    #         return len(self.models)

    # def average_models(self, rank):
    #     # Load state dict of requested rank
    #     state_dict_for_rank = self.models[0].state_dict()
    #     # Average all params
    #     for key in state_dict_for_rank:
    #         state_dict_for_rank[key] = sum(self.models[0].state_dict()[
    #                                        key] for r in self.models) / len(self.models)
    #     # Rewrite back state dict
    #     self.models[0].load_state_dict(state_dict_for_rank)


param_server = None
global_lock = Lock()


def get_parameter_server(rank, num_gpus=0):
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        # Add model for this rank
        # param_server.create_model_for_rank(rank, num_gpus)
        return param_server


def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers, hence it does not need to run a loop.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    logger.info("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    logger.info("RPC initialized! Running parameter server...")
    rpc.shutdown()
    logger.info("RPC shutdown on parameter server.")


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, rank, num_gpus=0,):
        super().__init__()
        self.num_gpus = num_gpus
        self.rank = rank
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(
                self.rank, num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(ParameterServer.get_param_rrefs,self.param_server_rref, self.rank)
        return remote_params

    def forward(self, x):
        model_output = remote_method(ParameterServer.forward, self.param_server_rref,self.rank,x)
        return model_output

    # def average_model_across_trainers(self):
    #     self.param_server_rref.rpc_sync().average_models(self.rank)


def run_training_loop(rank, world_size, num_gpus, train_loader, test_loader):
    # Runs the typical neural network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(rank=rank, num_gpus=num_gpus)
    # Wait for all nets on PS to be created, otherwise we could run
    # into race conditions during training.
    # num_created = net.param_server_rref.rpc_sync().get_num_models()
    # while num_created != world_size - 1:
    #     time.sleep(0.5)
    #     num_created = net.param_server_rref.rpc_sync().get_num_models()

    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
    for i, (data, target) in enumerate(train_loader):
        if TERMINATE_AT_ITER is not None and i == TERMINATE_AT_ITER:
            break
        if i % PS_AVERAGE_EVERY_N == 0:
            # Request server to update model with average params across all
            # trainers.
            logger.info(f"Rank {rank} averaging model across all trainers.")
#            net.average_model_across_trainers()
        with dist_autograd.context() as cid:
            model_output = net(data)
            target = target.to(model_output.device)
            loss = F.nll_loss(model_output, target)
            if i % TRAINER_LOG_INTERVAL == 0:
                logger.info(f"Rank {rank} training batch {i} loss {loss.item()}")

            wait_all_trainers(rank, world_size)
            if torch.__version__ > '1.4.0':
                dist_autogad.backward(cid, [loss])
            else:
                dist_autograd.backward([loss])
            wait_all_trainers(rank=rank, world_size=world_size)
            # Ensure that dist autograd ran successfully and gradients were
            # returned.
#            assert net.param_server_rref.rpc_sync().get_dist_gradients(cid) != {}
            print(f"Rank {rank} opt.step")
            if torch.__version__ > '1.4.0':
                opt.step(cid)
            else:
                opt.step()
            print(f"Rank {rank} done with step")
            wait_all_trainers(rank=rank, world_size=world_size)
            print(f"Rank {rank} done with wait all workers")

    logger.info("Training complete!")
#    logger.info("Getting accuracy....")
#    get_accuracy(test_loader, net)


def get_accuracy(test_loader, model):
    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
                          and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    logger.info(f"Accuracy {correct_sum / len(test_loader.dataset)}")


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    logger.info(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    logger.info(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, world_size, num_gpus, train_loader, test_loader)
    rpc.shutdown()

# --------- Launcher --------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="""Total number of participating processes. Should be the sum of
        master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="""Number of GPUs to use for training, currently supports between 0
         and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
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
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    processes = []
    world_size = args.world_size
    print("STATING")
    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    # else:
        # Get data to train on
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=32, shuffle=True)
        # start training worker on this node
        for rank in range(1, world_size):
            p = mp.Process(
                target=run_worker,
                args=(
                    rank,
                    world_size, args.num_gpus,
                    train_loader,
                    test_loader))
            p.start()
            processes.append(p)

    # for p in processes:
    #     p.join()
    # poll
    while True:
        for p in processes:
            if p.exitcode is not None and p.exitcode != 0:
                # kill rest of the processes
                for proc in processes:
                    if proc != p:
                        proc.kill()
                exit(1)
        if all(p.exitcode == 0 for p in processes):
            exit(0)