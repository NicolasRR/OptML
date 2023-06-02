import argparse
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import torch
import os
from tqdm import tqdm
from common import _get_model, create_testloader, LOSS_FUNC

DEFAULT_GRID_SIZE = 20
DEFAULT_BATCH_SIZE = 128


def set_weights(model, weights):
    weight_dict = {}
    idx = 0
    for key, param in model.state_dict().items():
        size = int(np.prod(param.shape))
        weight_dict[key] = torch.tensor(weights[idx : idx + size]).view(param.shape)
        idx += size

    model.load_state_dict(weight_dict)
    return model


def main(
    batch_size,
    subfolder,
    grid_size,
):
    
    classic_model="fashion_mnist_classic_0_100_0005_00_32_6_SGD_spe3_val_model.pt"
    classic_weights="fashion_mnist_classic_0_100_0005_00_32_6_SGD_spe3_val_weights.npy"

    async_m00_weights= "fashion_mnist_async_4_100_0005_00_32_6_SGD_spe3_val_weights.npy"
    async_m50_weights= "fashion_mnist_async_4_100_0005_05_32_6_SGD_spe3_val_weights.npy"
    async_m90_weights= "fashion_mnist_async_4_100_0005_09_32_6_SGD_spe3_val_weights.npy"
    async_m95_weights= "fashion_mnist_async_4_100_0005_095_32_6_SGD_spe3_val_weights.npy"
    async_m99_weights= "fashion_mnist_async_4_100_0005_099_32_6_SGD_spe3_val_weights.npy"
    async_alr_weights= "fashion_mnist_async_4_100_0001_09_32_6_ADAM_spe3_val_weights.npy"

    weights_paths = [classic_weights, async_m00_weights, async_m50_weights, async_m90_weights, async_m95_weights, async_m99_weights, async_alr_weights,]

    loaded_weights_np = []
    for wp in weights_paths:
        loaded_weights_np.append(np.load(wp))
        if "classic" in wp:
            print(f"Saved weights shape: {loaded_weights_np[-1].shape}")

    _min_w = 99999999
    _max_w = 0
    
    # perform PCA on the classic weights loaded_weights_np[0], remember the transformation to apply it to the other saved weights
    pca = PCA(n_components=2)
    reduced_weights = []
    for i, w in enumerate(loaded_weights_np):
        if i == 0:
            reduced_weights.append(pca.fit_transform(w))
        else:
            reduced_weights.append(pca.transform(w))

        _min = np.min(reduced_weights[-1])
        _max = np.max(reduced_weights[-1])

        if _min < _min_w:
            _min_w = _min
        if _max > _max_w:
            _max_w = _max

    _min_w = _min_w -1
    _max_w = _max_w +1

    grid_range_x = np.linspace(_min_w, _max_w, grid_size)
    grid_range_y = np.linspace(_min_w, _max_w, grid_size)

    xx, yy = np.meshgrid(grid_range_x, grid_range_y)

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    grid_weights = pca.inverse_transform(grid_points)

    for i, rw in enumerate(reduced_weights):
        if i != 0:
            loaded_weights_np[i] = pca.inverse_transform(rw)

    # Grid training
    loader = create_testloader(classic_model, batch_size)
    if "alt_model" in classic_model:
        model = _get_model(classic_model, LOSS_FUNC, alt_model=True)
    else:
        model = _get_model(classic_model, LOSS_FUNC, alt_model=False)

    model.load_state_dict(torch.load(classic_model))
    model.eval()

    grid_losses = []

    progress_bar = tqdm(
        total=len(grid_weights),
        desc="Computing loss of grid weights",
        unit="model_weights",
    )

    with torch.no_grad():
        for weights in grid_weights:
            model = set_weights(model, weights)

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

    # trajectories evaluationa
    trajectories_loss_reevaluted = []
    for w in loaded_weights_np:
        trajectory_loss_reevaluted = []
        progress_bar2 = tqdm(
            total=len(w),
            desc="Computing loss of trajectory weights",
            unit="model_weights",
        )

        with torch.no_grad():
            for weights in w:
                model = set_weights(model, weights)

                running_loss = 0.0
                for inputs, labels in loader:
                    outputs = model(inputs)
                    loss = LOSS_FUNC(outputs, labels)
                    running_loss += loss.item() * inputs.size(0)

                trajectory_loss_reevaluted.append(running_loss / len(loader.dataset))
                progress_bar2.update(1)
                progress_bar2.set_postfix(trajectory_loss=trajectory_loss_reevaluted[-1])

        progress_bar2.close()
        trajectories_loss_reevaluted.append(trajectory_loss_reevaluted)



    surface = go.Surface(
        x=xx,
        y=yy,
        z=grid_losses,
        opacity=0.8,
        name="grid point",
        coloraxis="coloraxis",
        colorscale="Viridis",
    )
    trajectory_names = [
        "Classic SGD m=0.0",
        "Async SGD m=0.0",
        "Async SGD m=0.50",
        "Async SGD m=0.90",
        "Async SGD m=0.95",
        "Async SGD m=0.99",
        "Async ADAM",
    ]
    trajectory_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta']  # Define more colors if you have more trajectories
    trajectories = []
    for i, (rw, tl) in enumerate(zip(reduced_weights, trajectories_loss_reevaluted)):
        traj = go.Scatter3d(
            x=rw[:, 0],
            y=rw[:, 1],
            z=tl,
            mode="markers+lines",
            line=dict(color=trajectory_colors[i % len(trajectory_colors)]),
            marker=dict(color=trajectory_colors[i % len(trajectory_colors)], size=5),
            name=trajectory_names[i],
        )
        trajectories.append(traj)

    layout = go.Layout(
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title=" Loss"),
        coloraxis=dict(colorbar=dict(title="Loss magnitude"), colorscale="Viridis"),
    )


    fig = go.Figure(data=[surface]+ trajectories, layout=layout)
    
    fig.update_layout(legend=dict(orientation="v", x=0, y=0.5))
    
    model_filename = os.path.basename(classic_model)
    model_basename, _ = os.path.splitext(model_filename)
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        output_file_path = os.path.join(
            subfolder, f"{model_basename}_loss_landscape_compare_{grid_size}.html"
        )
        pio.write_html(fig, output_file_path)
    else:
        output_file_path = f"{model_basename}_loss_landscape_compare_{grid_size}.html"
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

    if len(args.subfolder) > 0:
        print(f"Outputs will be saved to {args.subfolder}/")


    main(
        args.batch_size,
        args.subfolder,
        args.grid_size,
    )
