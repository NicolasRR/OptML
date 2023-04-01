import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class CNN_MNIST(nn.Module):  # for MNIST and Fashion MNIST
    def __init__(self):
        super(CNN_MNIST, self).__init__()
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


model_path = "mnist_sync_2_01_0001_00_64.pt"
model = CNN_MNIST()
model.load_state_dict(torch.load(model_path))

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=test_transform
)

test_loader = DataLoader(test_set, batch_size=100, shuffle=False)


def compute_digit_accuracy(model, data_loader):
    model.eval()
    correct = [0] * 10
    total = [0] * 10
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images, labels
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                total[label] += 1
                if predicted[i] == labels[i]:
                    correct[label] += 1

    accuracy = [correct[i] / total[i] for i in range(10)]
    return accuracy


digit_accuracy = compute_digit_accuracy(model, test_loader)
for digit, acc in enumerate(digit_accuracy):
    print(f"Accuracy for digit {digit}: {acc * 100:.2f}%")
