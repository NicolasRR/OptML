import argparse
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import torch
import torch.nn.functional as F
import os
import plotly.graph_objs as go
from tqdm import tqdm
from helpers import (
    CNN_MNIST,
    CNN_CIFAR10,
    CNN_CIFAR100,
    create_testloader,
)

DEFAULT_GRID_BORDER = 10
DEFAULT_GRID_SIZE = 10 
DEFAULT_BATCH_SIZE = 100

def main(batch_size, weights_path, model_path, subfolder, grid_size, grid_border):

    loader = create_testloader(model_path, batch_size)
    model = None
    if "mnist" in model_path:
        # print("Loading MNIST CNN\n")
        model = CNN_MNIST()
    elif "cifar100" in model_path:
        # print("Loading CIFAR100 CNN\n")
        model = CNN_CIFAR100()
    elif "cifar10" in model_path:
        # print("Loading CIFAR10 CNN\n")
        model = CNN_CIFAR10()
    else:
        # print("Dataset not supported\n")
        exit()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    weights_matrix_np= np.load(weights_path)

    print(f"Saved weights shape: {weights_matrix_np.shape}")

    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(weights_matrix_np)

    grid_range = np.linspace(-grid_border, grid_border, grid_size)
    xx, yy = np.meshgrid(grid_range, grid_range)

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_weights = pca.inverse_transform(grid_points)


    grid_losses = []

    progress_bar = tqdm(total= len(grid_weights), desc="Computing loss of grid weights", unit="model_weights")

    with torch.no_grad():
        for weights in grid_weights:
            # Convert the point to a dictionary with the same keys as the model's state_dict
            weight_dict = {}
            idx = 0
            for key, param in model.state_dict().items():
                size = np.prod(param.shape)
                weight_dict[key] = torch.tensor(weights[idx:idx+size]).view(param.shape)
                idx += size
            
            model.load_state_dict(weight_dict)
            
            running_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs, labels
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            grid_losses.append(running_loss / len(loader.dataset))
            progress_bar.update(1)
            progress_bar.set_postfix(grid_loss=grid_losses[-1])


    grid_losses = np.array(grid_losses).reshape(grid_size, grid_size)

    trajectory_loss_reevaluted = []

    progress_bar2 = tqdm(total= len(weights_matrix_np), desc="Computing loss of trajectory weights", unit="model_weights")

    with torch.no_grad():
        for weights in weights_matrix_np:
            # Convert the point to a dictionary with the same keys as the model's state_dict
            weight_dict = {}
            idx = 0
            for key, param in model.state_dict().items():
                size = np.prod(param.shape)
                weight_dict[key] = torch.tensor(weights[idx:idx+size]).view(param.shape)
                idx += size
            
            model.load_state_dict(weight_dict)
            
            running_loss = 0.0
            for inputs, labels in loader:
                inputs, labels = inputs, labels
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            trajectory_loss_reevaluted.append(running_loss / len(loader.dataset))
            progress_bar2.update(1)
            progress_bar2.set_postfix(trajectory_loss=trajectory_loss_reevaluted[-1])


    surface = go.Surface(x=xx, y=yy, z=grid_losses, opacity=0.8, name="grid point", coloraxis="coloraxis", colorscale="Viridis")

    colors = ["blue"] + ["red"] * (len(reduced_weights) - 2) + ["green"]
    sizes = [8] + [5] * (len(reduced_weights) - 2) + [8]

    trajectory = go.Scatter3d(
        x=reduced_weights[:, 0],
        y=reduced_weights[:, 1],
        z=trajectory_loss_reevaluted,
        mode='markers+lines',
        line=dict(color="red"),
        marker=dict(color=colors, size=sizes),
        name="Training Trajectory",
    )

    layout = go.Layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title=' Loss'
        ),
        coloraxis=dict(colorbar=dict(title="Loss magnitude"), colorscale="Viridis"), 
    )

    fig = go.Figure(data=[surface, trajectory], layout=layout)
    #fig.show()
    model_filename = os.path.basename(model_path)
    model_basename, _ = os.path.splitext(model_filename)
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        output_file_path = os.path.join(subfolder, f"{model_basename}_loss_landscape.html")
        pio.write_html(fig, output_file_path) 
    else:
        output_file_path = f"{model_basename}_loss_landscape.html"
        pio.write_html(fig, output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing models")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size to forward the test dataset.""",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="""grid_size^2 amount of points to populate the 2D space to evaluate the loss.""",
    )
    parser.add_argument(
        "--grid_border",
        type=int,
        default=None,
        help="""The grid will be created between [-grid_border, grid_border]x[-grid_border, grid_border].""",
    )
    parser.add_argument(
        "weights_path", type=str, help="""Weights of the trained model."""
    )
    parser.add_argument(
        "model_path", type=str, help="""Path of the model."""
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="""Subfolder name where the test results and plots will be saved.""",
    )

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"\nUsing default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(test set)]")
        exit()
    
    if args.grid_size is None:
        args.grid_size = DEFAULT_GRID_SIZE
        print(f"Using default grid_size: {DEFAULT_GRID_SIZE}")
    elif args.grid_size < 1:
        print("Forbidden value !!! grid_size must be > 1")
        exit()

    if args.grid_border is None:
        args.grid_border = DEFAULT_GRID_BORDER
        print(f"Using default grid_border: {DEFAULT_GRID_BORDER}")
    elif args.grid_border <= 0:
        print("Forbidden value !!! grid_border must be <= 0")
        exit()

    if len(args.subfolder) > 0:
        print(f"Outputs will be saved to {args.subfolder}/")

    if len(args.weights_path) == 0:
        print("Missing weights path !!!")
        exit() 

    if len(args.model_path) == 0:
        print("Missing model path !!!")
        exit() 

    main(
        args.batch_size, 
        args.weights_path,
        args.model_path,
        args.subfolder,
        args.grid_size,
        args.grid_border,
    )
