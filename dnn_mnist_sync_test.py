import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

DEFAULT_BATCH_SIZE = 1000

#if train architecture changes, change the Net class
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

def main(model_path, batch_size):

    # Load the saved model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('data/', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model on the test dataset
    test_loss = 0
    correct = 0
    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            targets.extend(target.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print("\n")
    print(f"Average accuracy: {accuracy*100:.2f} % ({correct}/{len(test_loader.dataset)})")
    print(f'Average loss: {test_loss:.4f}')
    per_class_accuracy = [accuracy_score(np.array(targets) == i, np.array(predictions) == i) for i in range(10)]

    print('Per-class metrics:')
    print('Class\tAccuracy')
    for i in range(10):
        print(f'{i}\t{per_class_accuracy[i]:.4f}')

    report = classification_report(targets, predictions)
    print('\nClassification report:')
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Asynchronous-parameter-Server RPC based training")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="""Path of the trained model.""")
    
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    if args.path is None:
        print("Missing model path !!!")
        exit()

    main(args.path, args.batch_size)