import argparse
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import torch
import os
import plotly.graph_objs as go
from tqdm import tqdm
from common import _get_model, create_testloader, LOSS_FUNC

DEFAULT_GRID_SIZE = 20
DEFAULT_BATCH_SIZE = 100
DEFAULT_GRID_WARNING = 10


def set_weights(model, flat_weights):
    idx = 0
    state_dict = model.state_dict()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params == len(
        flat_weights
    ), f"Number of model parameters ({total_params}) doesn't match length of weights array ({len(flat_weights)})"

    for key, param in state_dict.items():
        if param.requires_grad:
            param_size = torch.prod(torch.tensor(param.shape)).item()
            param_flat = flat_weights[idx : idx + param_size]
            state_dict[key] = param_flat.view(param.shape)
            idx += param_size

    model.load_state_dict(state_dict)


def main(
    batch_size,
    weights_path,
    model_path,
    subfolder,
    grid_size,
    grid_border=None,
):
    loader = create_testloader(model_path, batch_size)
    if "alt_model" in model_path:
        model = _get_model(model_path, LOSS_FUNC, alt_model=True)
    else:
        model = _get_model(model_path, LOSS_FUNC, alt_model=False)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    weights_matrix_np = np.load(weights_path)

    print(f"Saved weights shape: {weights_matrix_np.shape}")

    pca = PCA(n_components=2)
    reduced_weights = pca.fit_transform(weights_matrix_np)
    print(reduced_weights.shape)
    max_reduced_weight = np.max(reduced_weights, axis = 0)
    min_reduced_weight = np.min(reduced_weights, axis = 0)
    print(
        f"Norm of largest weights in the PCA space: {max_reduced_weight, min_reduced_weight}"
    )
    grid_center = np.array([np.mean(max_reduced_weight), np.mean(min_reduced_weight)])
    if grid_border is None:
        grid_border = 4*np.max(max_reduced_weight-min_reduced_weight)
    
    else:
        # Check if the grid is too small
        if np.max(max_reduced_weight-min_reduced_weight) > grid_border:
            print(
                f"Warning: The grid might be too small. The maximum absolute value of the reduced weights is outside the grid border ({grid_border})."
            )

        # Check if the grid is too big
        if np.abs(np.max(max_reduced_weight-min_reduced_weight) - grid_border) > DEFAULT_GRID_WARNING:
            print(
                f"Warning: The grid might be too big. The distance from the maximum absolute value of the reduced weights to the grid border ({grid_border}) is greater than 10."
            )

    grid_range = np.linspace(-grid_border+grid_center[0], grid_border+grid_center[1], grid_size)
    xx, yy = np.meshgrid(grid_range, grid_range)

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_weights = pca.inverse_transform(grid_points)

    grid_losses = []

    progress_bar = tqdm(
        total=len(grid_weights),
        desc="Computing loss of grid weights",
        unit="model_weights",
    )

    with torch.no_grad():
        for weights in grid_weights:
            weights_torch = torch.tensor(weights).float()
            set_weights(model, weights_torch)

            running_loss = 0.0
            for inputs, labels in loader:
                outputs = model(inputs)
                loss = LOSS_FUNC(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            grid_losses.append(running_loss / len(loader.dataset))
            progress_bar.update(1)
            progress_bar.set_postfix(grid_loss=grid_losses[-1])

    progress_bar.close()

    grid_losses = np.array(grid_losses).reshape(grid_size, grid_size)

    trajectory_loss_reevaluted = []

    progress_bar2 = tqdm(
        total=len(weights_matrix_np),
        desc="Computing loss of trajectory weights",
        unit="model_weights",
    )

    with torch.no_grad():
        for weights in weights_matrix_np:
            weights_torch = torch.tensor(weights).float()
            set_weights(model, weights_torch)

            running_loss = 0.0
            for inputs, labels in loader:
                outputs = model(inputs)
                loss = LOSS_FUNC(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            trajectory_loss_reevaluted.append(running_loss / len(loader.dataset))
            progress_bar2.update(1)
            progress_bar2.set_postfix(trajectory_loss=trajectory_loss_reevaluted[-1])

    progress_bar2.close()

    surface = go.Surface(
        x=xx,
        y=yy,
        z=grid_losses,
        opacity=0.8,
        name="grid point",
        coloraxis="coloraxis",
        colorscale="Viridis",
    )

    colors = ["blue"] + ["red"] * (len(reduced_weights) - 2) + ["green"]
    sizes = [8] + [5] * (len(reduced_weights) - 2) + [8]

    trajectory = go.Scatter3d(
        x=reduced_weights[:, 0],
        y=reduced_weights[:, 1],
        z=trajectory_loss_reevaluted,
        mode="markers+lines",
        line=dict(color="red"),
        marker=dict(color=colors, size=sizes),
        name="Training Trajectory",
    )

    layout = go.Layout(
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title=" Loss"),
        coloraxis=dict(colorbar=dict(title="Loss magnitude"), colorscale="Viridis"),
    )

    fig = go.Figure(data=[surface, trajectory], layout=layout)
    # fig.show()
    model_filename = os.path.basename(model_path)
    model_basename, _ = os.path.splitext(model_filename)
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        output_file_path = os.path.join(
            subfolder, f"{model_basename}_loss_landscape.html"
        )
        pio.write_html(fig, output_file_path)
    else:
        output_file_path = f"{model_basename}_loss_landscape.html"
        pio.write_html(fig, output_file_path)

    print(f"Saved 3D figure at: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computing the loss landscape with the training trajectory, please model path (.pt) first, weights (.npy) second"
    )
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
        help="""The grid will be created between [-grid_border+center, grid_border+center]x[-grid_border+center, grid_border+center].""",
    )
    parser.add_argument("model_path", type=str, help="""Path of the model.""")
    parser.add_argument(
        "weights_path", type=str, help="""Weights of the trained model."""
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

    if not(args.grid_border is None) and args.grid_border <= 0:
        print("Forbidden value !!! grid_border must be > 0")
        exit()

    if len(args.subfolder) > 0:
        print(f"Outputs will be saved to {args.subfolder}/")

    if len(args.weights_path) == 0:
        print("Missing weights path !!!")
        exit()

    if len(args.model_path) == 0:
        print("Missing model path !!!")
        exit()

    if not args.weights_path.endswith(".npy"):
        print(
            "weights_path should be a .npy file, the order should be: model first weights second"
        )
        exit()

    if not args.model_path.endswith(".pt"):
        print(
            "model_path should be a .pt file, the order should be: model first weights second"
        )
        exit()

    main(
        args.batch_size,
        args.weights_path,
        args.model_path,
        args.subfolder,
        args.grid_size,
        args.grid_border,
    )
