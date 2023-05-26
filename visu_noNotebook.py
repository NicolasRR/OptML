import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import Subset


# Define the LeNet5 model
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# TRAIN THE MODEL ON THE DATASET
# Load MNIST data
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
# Define the size of the subset
subset_size = 1000
indices = torch.randperm(len(full_dataset))[:subset_size]
subset_dataset = Subset(full_dataset, indices)
# Create the DataLoader
data_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)

# Instantiate the model
model = LeNet5()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
# Optimizer definition
optimizer = optim.Adam(model.parameters(), lr=0.001)
try:
    # Load the saved model
    model_weights_path = 'models_weights/lenet5_mnist_weights.pth'
    saved_weights = torch.load(model_weights_path)
    # Create a new model and load the saved weights
    model = LeNet5()
    model.load_state_dict(saved_weights)

except:
    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, labels in data_loader:
        
            labels = labels

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

    model_weights_path = 'models_weights/lenet2_mnist_weights_2.pth'
    torch.save(model.state_dict(), model_weights_path)
    
    import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors

class RandomCoordinates(object):
    def __init__(self, origin):
        self.origin_ = origin
        self.v0_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )
        self.v1_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]


def normalize_weights(weights, origin):
    return [
        w * np.linalg.norm(wc) / np.linalg.norm(w)
        for w, wc in zip(weights, origin)
    ]
    
class LossSurface(object):
    def __init__(self, model, inputs, labels, criterion):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.criterion = criterion

    def compile(self, range_val, points, coords):
        a_grid = np.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        b_grid = np.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        original_weights = [w.clone() for w in self.model.parameters()]
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                new_weights = coords(a, b)
                
                for param, new_weight in zip(self.model.parameters(), new_weights):
                    param.data.copy_(new_weight)

                pred = self.model(self.inputs)
                loss = self.criterion(pred, self.labels)
                loss_grid[j, i] = loss.item()
        # Restore original model weights
        for param, original_weight in zip(self.model.parameters(), original_weights):
            param.data.copy_(original_weight)

        self.a_grid = a_grid
        self.b_grid = b_grid
        self.loss_grid = loss_grid

    def plot(self, range_val=1.0, points=24, levels=20, ax=None, **kwargs):
        xs = self.a_grid
        ys = self.b_grid
        zs = self.loss_grid
        if ax is None:
            _, ax = plt.subplots(**kwargs)
            ax.set_title("The Loss Surface")
            ax.set_aspect("equal")
        # Set Levels
        min_loss = zs.min()
        max_loss = zs.max()
        levels = np.exp(
            np.linspace(
                np.log(min_loss), np.log(max_loss), num=levels
            )
        )
        CS = ax.contour(xs, ys, zs)
        ax.clabel(CS, inline=True, fontsize=10)
        return ax

# start taking making the grid
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
model_weight = Params2Vec(model.parameters())
model_weight_np = model_weight.detach().numpy()

coords = RandomCoordinates(model_weight_np)
inputs, labels  = next(iter(data_loader))


loss_surface = LossSurface(model, inputs, labels, criterion=criterion)
loss_surface.compile(range_val = 1,points=30, coords=coords)


# Save loss surface data 
loss_surface.loss_grid.shape
# Look at loss surface
plt.figure(dpi=100)
loss_surface.plot()