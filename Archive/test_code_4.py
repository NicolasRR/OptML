import torch
import torchvision
import torchvision.transforms as transforms

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

# Compute the mean and standard deviation
data = next(iter(trainloader))[0]
mean = data.mean()
std = data.std()

print("Mean:", mean.item())
print("Standard Deviation:", std.item())