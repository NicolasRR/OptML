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
import copy
from torch.optim.swa_utils import SWALR


class Net(nn.Module):
    def __init__(self, num_input=1):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(num_input, 32, 3, 1)
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
    """
    This class contains the parameters and updates them at every iteration. To do so it performs thread locking in order to avoid memory issues. 

    """

    def __init__(self, delay, ntrainers, lr, momentum, logger, mode):
        self.model = Net()
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum = momentum)
        self.losses = np.array([])
        self.norms = np.array([])
        for _,p in enumerate(self.model.parameters()):
            p.grad = torch.zeros_like(p)
        self.delay = delay  
        self.logger = logger
        self.mode = mode
        if mode == "scheduler":
            self.logger.info("Creating a Learning rate scheduler")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
        if mode == "swa":
            self.logger.info("Creating a SWA scheduler")
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)
            self.swa_start = 2
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.05)
        elif mode == "compensation":
            self.logger.info("Using delay compensation")
            self.backups = [[torch.randn_like(param, requires_grad=False) for param in self.model.parameters()] for _ in range(ntrainers)]
        self.epochs = np.zeros(ntrainers, dtype=int)




    def get_model(self, id):
        if self.mode == "compensation":
            self.backups[id-1] = [param for param in self.model.parameters()]
        return self.model


    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads,id, loss, epoch):
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
            for i,(p, g) in enumerate(zip(self.model.parameters(), grads)):
                if p.grad is not None:
                    if self.mode == "compensation":
                        p.grad += g+0.5*g*g*(p-self.backups[id-1][i])
                    else:
                        p.grad += g
                    n = p.grad.norm(2).item() ** 2
                    norm += n
                else:
                    self.logger.debug(f"None p.grad detected for the update")
            self.logger.debug(f"Loss is {self.losses[-1]}")
            self.norms = np.append(self.norms, norm**(0.5))
            self.logger.debug(f"The norm of the gradient is {self.norms[-1]}")
            self.loss = np.array([])
            self.optimizer.step()
            if self.epochs[id-1] < epoch:
                self.epochs[id-1] += 1
                if self.mode == "swa" and epoch>self.swa_start:
                    self.logger.debug("SWA scheduler update")
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                elif self.mode == "swa" or self.mode == "scheduler":
                    self.logger.debug("Scheduler Upate")
                    self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=False)
            fut.set_result(self.model)
            self.logger.debug("PS updated model")
            self.future_model = torch.futures.Future()

        return fut


class Trainer(object):
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
            for (inputs,labels) in tqdm(self.dataloader, position=self.name, leave=False, desc=f"Epoch {i}"):
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
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.parameters()], self.name, loss, i),
            )


def run_trainer(ps_rref, dataloader, name, epochs, output_folder):
    """
    Individually initialize the trainers with their loggers and start training
    """
    tlogger = logging.getLogger(f"t{name}_logger")
    tlogger.setLevel(logging.DEBUG)
    fht = logging.FileHandler(os.path.join(output_folder,f"t{name}.log"), mode="w")
    fht.setLevel(logging.DEBUG)
    fht.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    tlogger.addHandler(fht)
    trainer = Trainer(ps_rref, dataloader, name, epochs, tlogger)
    tlogger.debug(f"Run trainer {name}")
    trainer.train()


def run_ps(trainers, output_folder, dataset, batch_size, epochs, delay, learning_rate, momentum, model_accuracy, save_model, mode):
    """
    This is the main function which launches the training of all the slaves
    """
    pslogger = logging.getLogger("ps_logger")
    pslogger.setLevel(logging.DEBUG)

    # File Handler for debug messages and keeping track
    fh = logging.FileHandler(os.path.join(output_folder,"ps.log"), mode="w")
    fh.setLevel(logging.DEBUG)
    # Console Handler for high level infor
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    pslogger.addHandler(ch)
    pslogger.addHandler(fh)
    pslogger.info("Start training")

    ps_rref = rpc.RRef(BatchUpdateParameterServer(delay, len(trainers),learning_rate, momentum, pslogger, mode))
    futs = []
    
    # Random partition of the classes for each trainer
    classes = np.unique(dataset.targets)
    np.random.shuffle(classes)
    chunks = np.floor(len(classes)/len(trainers)).astype(int)
    targets = [classes[i*chunks: (i+1)*chunks] for i in range(0, len(trainers))]
    

    for id, trainer in enumerate(trainers):
        pslogger.info(f"Trainer {id+1} working with classes {targets[id]}")

        # Dataset slicing for each worker
        d = copy.deepcopy(dataset)
        idx = np.isin(d.targets,targets[id])
        d.data = d.data[idx]
        d.targets = d.targets[idx]
        dataloader = torch.utils.data.DataLoader(d, 
                                batch_size=batch_size, 
                                shuffle=True)

        futs.append(
            rpc.rpc_async(trainer, run_trainer, args=(ps_rref,dataloader, id+1,epochs, output_folder,))
        )

    torch.futures.wait_all(futs)
    ps = ps_rref.to_here()
    
    np.savetxt(os.path.join(output_folder,"loss.txt"), ps.losses)
    np.savetxt(os.path.join(output_folder,"norms.txt"), ps.norms)

    pslogger.info("Finish training")

    
    if model_accuracy:
        train_loader_full = torch.utils.data.DataLoader(dataset, 
                                batch_size=batch_size*10, 
                                shuffle=True)
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
        filename = f"mnist_async_{learning_rate}_{momentum}_{batch_size}_numclasses_{len(targets)*chunks}.pt"
        torch.save(ps_rref.to_here().model.state_dict(), os.path.join(output_folder,filename))
        print(f"Model saved: {filename}")
    logging.shutdown()



def run(rank, world_size,  output_folder, dataset, batch_size, epochs, delay, learning_rate, momentum, model_accuracy, save_model, mode):
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
        run_ps([f"trainer{r}" for r in range(1, world_size)], output_folder, dataset, batch_size, epochs, delay, learning_rate, momentum, model_accuracy, save_model, mode)

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
    parser.add_argument(
        "--output",
        type=str,
        default="./results/dnn_mnist_async",
        help="""Output folder.""")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="""Number of epochs"""),
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="""delay in seconds""")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="""Batch size""")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="""Learning rate""")
    parser.add_argument(
        "--momemtum",
        type=float,
        default=0.9,
        help="""Momentum""")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="""Dataset amongst: {mnist, cifar10, cifar100, fmnist}""")
    parser.add_argument(
        '-m', help="Model accuracy",
        action='store_false')
    parser.add_argument(
        '-ns', help="Save model",
        action='store_true')
    parser.add_argument(
        '--mode', 
        type=str,
        default="normal",
        help="""Choose amongst {compensation, scheduler swa}, scheduler indicates to only use a learning rate scheduler""")
    args = parser.parse_args()
    output_folder = args.output
    epochs = args.epochs
    delay = args.delay
    save_model = not(args.ns)
    model_accuracy = args.m 
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = args.m
    dataset_name = args.dataset
    mode = args.mode

    if dataset_name == "mnist":
        print("Using MNIST dataset")
        dataset = torchvision.datasets.MNIST('./../data/mnist_data', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
    
    elif dataset_name == "cifar10":
        print("Using CIFAR10 dataset")
        dataset = torchvision.datasets.CIFAR10('./../data/cifar10', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
    elif dataset_name == "cifar100":
        print("Using CIFAR100 dataset")
        dataset = torchvision.datasets.CIFAR100('./../data/cifar100', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
    elif dataset_name == "fmnist":
        print("Using FashionMNIST dataset")
        dataset = torchvision.datasets.FashionMNIST('./../data/fmnist', 
                                                download=True, 
                                                train=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
                                                ]))
                                                
    else:
        print("You must specify a dataset amongst: {mnist, cifar10, cifar100, fmnist}")
        exit()

    if not(os.path.exists(output_folder)):
        os.mkdir(output_folder)
    
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    world_size = args.world_size
    

    mp.spawn(run, args=(world_size, output_folder, dataset, batch_size, epochs, delay, learning_rate, momentum, model_accuracy, save_model, mode,), nprocs=world_size, join=True)
