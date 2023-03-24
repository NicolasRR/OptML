import argparse
import os
import torch
import torch.distributed.rpc as rpc
from torch import nn
from torchvision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch RPC MNIST Example')
    parser.add_argument('--master_addr', type=str, default='localhost',
                        help='Address of master node')
    parser.add_argument('--master_port', type=str, default='29500',
                        help='Port of master node')
    return parser.parse_args()

class MNISTTrainer(nn.Module):
    def __init__(self, digit):
        super().__init__()
        self.digit = digit
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

    def train(self, data):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        # Get remote reference to parameter server
        ps_rref = rpc.remote("trainer0", ParameterServer)
        num_correct = 0
        for epoch in range(1):
            epoch_loss = 0.0
            for i in range(len(data)):
                x, y = data[i]
                x = x.view(-1, 784)
                y = torch.tensor([y == self.digit], dtype=torch.float32).unsqueeze(0)

                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                # Send gradients to parameter server
                ps_rref.rpc_async().update_params(self.model)
                optimizer.step()

                epoch_loss += loss.item()
                pred = output >= 0.5
                num_correct += pred.eq(y.view_as(pred)).sum().item()

                if (i + 1) % 100 == 0:
                    print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.4f}")

            epoch_loss /= len(data)
            accuracy = num_correct / len(data)
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList([
            nn.Parameter(torch.zeros(128, 784)),
            nn.Parameter(torch.zeros(128)),
            nn.Parameter(torch.zeros(1, 128)),
            nn.Parameter(torch.zeros(1))
        ])

    def update_params(self, params):
        for p1, p2 in zip(self.params.parameters(), params.parameters()):
            p1.data += p2.grad

def run_trainer(rank, world_size):
    # Disable shared memory transport
    options = rpc.TensorPipeRpcBackendOptions(
        _transports=["ibv", "uv"],
        _channels=["basic"]
    )

    rpc.init_rpc(f"trainer{rank}", rank=rank, world_size=world_size,
                 rpc_backend_options=options)

    if rank == 0:
        # Rank 0 is the parameter server
        ps = ParameterServer()
    else:
        # Other ranks are trainers
        train_data = datasets.MNIST('./data', train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                    ]))

        # Split the data by digit
        digit_data = [[] for _ in range(10)]
        for i in range(len(train_data)):
            x, y = train_data[i]
            digit_data[y].append((x, y))

        trainer = MNISTTrainer(rank - 1)
        trainer.train(digit_data[rank - 1])

    rpc.shutdown()

if __name__ == '__main__':
    args = parse_args()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    world_size = 11
    torch.multiprocessing.spawn(run_trainer, args=(world_size,), nprocs=world_size)