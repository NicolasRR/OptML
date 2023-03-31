import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST
from torch.utils.data import DataLoader

# Data transformation for CIFAR-10 and CIFAR-100
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Data transformation for MNIST and Fashion MNIST
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

# Load CIFAR-100 dataset
cifar100_train = CIFAR100(root='./data', train=True, download=True, transform=transform_cifar)
cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform_cifar)

# Load Fashion MNIST dataset
fashion_mnist_train = FashionMNIST(root='./data', train=True, download=True, transform=transform_mnist)
fashion_mnist_test = FashionMNIST(root='./data', train=False, download=True, transform=transform_mnist)

# Load Fashion MNIST dataset
mnist_train = MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test = MNIST(root='./data', train=False, download=True, transform=transform_mnist)

# Create DataLoaders
batch_size = 100
train_loader_cifar10 = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader_cifar10 = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

train_loader_cifar100 = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
test_loader_cifar100 = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

train_loader_fashion_mnist = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
test_loader_fashion_mnist = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)

train_loader_mnist = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader_mnist = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)







