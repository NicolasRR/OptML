import argparse
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import torch
import os
from tqdm import tqdm
from common import _get_model, create_testloader, set_seeds, LOSS_FUNC

DEFAULT_GRID_SIZE = 50
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
    path_grid_losses,
    path_grid_xx,
    path_grid_yy,
    grid_save=True,
):
    set_seeds()

    # Hardcoded paths
    model_path = "Results_AsyncMomentum/fashion_mnist_async_4_100_0005_00_32_6_SGD_spe3_val_model.pt"

    async_m00_weights = "Results_AsyncMomentum/fashion_mnist_async_4_100_0005_00_32_6_SGD_spe3_val_weights.npy"
    async_m50_weights = "Results_AsyncMomentum/fashion_mnist_async_4_100_0005_05_3261_SGD_spe3_val_weights.npy"
    async_m90_weights = "Results_AsyncMomentum/fashion_mnist_async_4_100_0005_09_32_6_SGD_spe3_val_weights.npy"
    async_m95_weights = "Results_AsyncMomentum/fashion_mnist_async_4_100_0005_095_32_6_SGD_spe3_val_weights.npy"

    weights_paths = [
        async_m00_weights,
        async_m50_weights,
        async_m90_weights,
        async_m95_weights,
    ]

    loaded_weights_np = []
    for wp in weights_paths:
        loaded_weights_np.append(np.load(wp))
        if "fashion_mnist_async_4_100_0005_00" in wp:
            print(f"Saved weights shape: {loaded_weights_np[-1].shape}")

    # perform PCA, remember the transformation to apply it to the other saved weights
    pca = PCA(n_components=2)

    # Convert the 3D list to a NumPy array
    matrix_array = np.array(loaded_weights_np)
    # Reshape the array to combine the matrices horizontally
    combined_matrix = np.vstack(matrix_array)
    pca.fit(combined_matrix)

    reduced_weights = []
    _min_w = 99999999
    _max_w = 0
    for w in loaded_weights_np:
        reduced_weights.append(pca.transform(w))

        _min = np.min(reduced_weights[-1])
        _max = np.max(reduced_weights[-1])

        if _min < _min_w:
            _min_w = _min
        if _max > _max_w:
            _max_w = _max

    _min_w = _min_w - 1
    _max_w = _max_w + 1

    if path_grid_losses is None and path_grid_xx is None and path_grid_yy is None:
        grid_range_x = np.linspace(_min_w, _max_w, grid_size)
        grid_range_y = np.linspace(_min_w, _max_w, grid_size)

        xx, yy = np.meshgrid(grid_range_x, grid_range_y)

        grid_points = np.column_stack((xx.ravel(), yy.ravel()))
        grid_weights = pca.inverse_transform(grid_points)

    for i, rw in enumerate(reduced_weights):
        loaded_weights_np[i] = pca.inverse_transform(rw)

    # loading model
    loader = create_testloader(model_path, batch_size)
    if "alt_model" in model_path:
        model = _get_model(model_path, LOSS_FUNC, alt_model=True)
    else:
        model = _get_model(model_path, LOSS_FUNC, alt_model=False)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Grid training
    if path_grid_losses is None and path_grid_xx is None and path_grid_yy is None:
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

        if grid_save:
            model_filename = os.path.basename(model_path)
            model_basename, _ = os.path.splitext(model_filename)

            if len(subfolder) > 0:
                if not os.path.exists(subfolder):
                    os.makedirs(subfolder)
                np.save(
                    os.path.join(
                        subfolder, f"{model_basename}_grid_losses_{grid_size}.npy"
                    ),
                    grid_losses,
                )
                np.save(
                    os.path.join(
                        subfolder, f"{model_basename}_grid_xx_{grid_size}.npy"
                    ),
                    xx,
                )
                np.save(
                    os.path.join(
                        subfolder, f"{model_basename}_grid_yy_{grid_size}.npy"
                    ),
                    yy,
                )
                print(
                    f"Saved grid losses to: {os.path.join(subfolder, f'{model_basename}_grid_losses_{grid_size}.npy')}"
                )
                print(
                    f"Saved grid xx to: {os.path.join(subfolder, f'{model_basename}_grid_xx_{grid_size}.npy')}"
                )
                print(
                    f"Saved grid yy to: {os.path.join(subfolder, f'{model_basename}_grid_yy_{grid_size}.npy')}"
                )
            else:
                np.save(f"{model_basename}_grid_losses_{grid_size}.npy", grid_losses)
                np.save(f"{model_basename}_grid_xx_{grid_size}.npy", xx)
                np.save(f"{model_basename}_grid_yy_{grid_size}.npy", yy)
                print(
                    f"Saved grid losses to: {f'{model_basename}_grid_losses_{grid_size}.npy'}"
                )
                print(
                    f"Saved grid xx to: {f'{model_basename}_grid_xx_{grid_size}.npy'}"
                )
                print(
                    f"Saved grid yy to: {f'{model_basename}_grid_yy_{grid_size}.npy'}"
                )

    else:
        xx = np.load(path_grid_xx)
        yy = np.load(path_grid_yy)
        grid_losses = np.load(path_grid_losses)
        print("Loadded grid_losses, grid_xx and grid_yy")

    # trajectories evaluation
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
                progress_bar2.set_postfix(
                    trajectory_loss=trajectory_loss_reevaluted[-1]
                )

        progress_bar2.close()
        trajectories_loss_reevaluted.append(trajectory_loss_reevaluted)

    if grid_save:
        if len(subfolder) > 0:
            np.save(
                os.path.join(
                    subfolder,
                    f"{model_basename}_trajectories_losses_{grid_size}.npy",
                ),
                np.vstack(
                    [
                        reduced_weights[0][:, 0],
                        reduced_weights[0][:, 1],
                        trajectories_loss_reevaluted[0],
                        reduced_weights[1][:, 0],
                        reduced_weights[1][:, 1],
                        trajectories_loss_reevaluted[1],
                        reduced_weights[2][:, 0],
                        reduced_weights[2][:, 1],
                        trajectories_loss_reevaluted[2],
                        reduced_weights[3][:, 0],
                        reduced_weights[3][:, 1],
                        trajectories_loss_reevaluted[3],
                    ]
                ),
            )
            print(
                f"Saved trajectories losses to: {os.path.join(subfolder, f'{model_basename}_trajectories_losses_{grid_size}.npy')}"
            )
        else:
            np.save(
                f"{model_basename}_trajectories_losses_{grid_size}.npy",
                np.vstack(
                    [
                        reduced_weights[0][:, 0],
                        reduced_weights[0][:, 1],
                        trajectories_loss_reevaluted[0],
                        reduced_weights[1][:, 0],
                        reduced_weights[1][:, 1],
                        trajectories_loss_reevaluted[1],
                        reduced_weights[2][:, 0],
                        reduced_weights[2][:, 1],
                        trajectories_loss_reevaluted[2],
                        reduced_weights[3][:, 0],
                        reduced_weights[3][:, 1],
                        trajectories_loss_reevaluted[3],
                    ]
                ),
            )
            print(
                f"Saved trajectories losses to: {f'{model_basename}_trajectories_losses_{grid_size}.npy'}"
            )

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
        "Async SGD m=0.0",
        "Async SGD m=0.50",
        "Async SGD m=0.90",
        "Async SGD m=0.95",
    ]

    trajectory_colors = [
        "orange",
        "magenta",
        "yellow",
        "cyan",
    ]
    trajectories = []
    labels = (
        ["Start Point"]
        + ["Trajectory Point"] * (len(trajectory_loss_reevaluted) - 2)
        + ["End Point"]
    )
    sizes = [8] + [5] * (len(trajectory_loss_reevaluted) - 2) + [8]
    for i, (rw, tl) in enumerate(zip(reduced_weights, trajectories_loss_reevaluted)):
        traj = go.Scatter3d(
            x=rw[:, 0],
            y=rw[:, 1],
            z=tl,
            mode="markers+lines",
            line=dict(color=trajectory_colors[i % len(trajectory_colors)]),
            marker=dict(
                color=trajectory_colors[i % len(trajectory_colors)], size=sizes
            ),
            name=trajectory_names[i],
            text=[
                f"{label}<br>Loss: {loss}"
                for label, loss in zip(labels, trajectories_loss_reevaluted[i])
            ],
            hovertemplate="%{text}",
        )
        trajectories.append(traj)

    layout = go.Layout(
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title=" Loss",
            zaxis=dict(type="log"),
        ),
        coloraxis=dict(
            colorbar=dict(title="Loss magnitude"),
            colorscale="Viridis",
            cmin=np.log10(grid_losses.min()),
            cmax=np.log10(grid_losses.max()),
        ),
    )

    fig = go.Figure(data=[surface] + trajectories, layout=layout)

    fig.data[0].update(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )

    _min_x, _min_y = np.unravel_index(np.argmin(grid_losses), grid_losses.shape)
    min_point = go.Scatter3d(
        x=[xx[_min_x, _min_y]],
        y=[yy[_min_x, _min_y]],
        z=[grid_losses[_min_x, _min_y]],
        mode="markers",
        marker=dict(
            size=5,
            color="red",
        ),
        name="global minimum",
    )

    fig.add_trace(min_point)

    fig.update_layout(legend=dict(orientation="v", x=0, y=0.5))

    fig2 = go.Figure(
        data=[
            go.Contour(
                x=xx.flatten(),
                y=yy.flatten(),
                z=np.log10(grid_losses.flatten()),
                colorscale="Viridis",
                name="grid point",
                coloraxis="coloraxis",
            )
        ]
    )

    trajectories = []
    for i, rw in enumerate(reduced_weights):
        traj = go.Scatter(
            x=rw[:, 0],
            y=rw[:, 1],
            mode="markers+lines",
            line=dict(color=trajectory_colors[i % len(trajectory_colors)]),
            marker=dict(
                color=trajectory_colors[i % len(trajectory_colors)], size=sizes
            ),
            name=trajectory_names[i],
            text=[
                f"{label}<br>Loss: {loss}"
                for label, loss in zip(labels, trajectories_loss_reevaluted[i])
            ],
            hovertemplate="%{text}",
        )
        trajectories.append(traj)

    for traj in trajectories:
        fig2.add_trace(traj)

    fig2.update_layout(
        title="Contour Plot with trajectories Projection",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend=dict(orientation="v", x=1, y=1),
    )

    min_point = go.Scatter(
        x=[xx[_min_x, _min_y]],
        y=[yy[_min_x, _min_y]],
        mode="markers",
        marker=dict(
            size=7,
            color="red",
            symbol="star",
        ),
        name="global minimum",
        text=[str(np.min(grid_losses))],
        hovertemplate="Loss: %{text}",
    )

    fig2.add_trace(min_point)

    model_filename = os.path.basename(model_path)
    model_basename, _ = os.path.splitext(model_filename)
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        output_file_path = os.path.join(
            subfolder, f"{model_basename}_loss_landscape_compare_{grid_size}.html"
        )
        output_file_path_2 = os.path.join(
            subfolder, f"{model_basename}_contour_landscape_compare_{grid_size}.html"
        )
        pio.write_html(fig, output_file_path)
        pio.write_html(fig2, output_file_path_2)
    else:
        output_file_path = f"{model_basename}_loss_landscape_compare_{grid_size}.html"
        output_file_path_2 = (
            f"{model_basename}_contour_landscape_compare_{grid_size}.html"
        )
        pio.write_html(fig, output_file_path)
        pio.write_html(fig2, output_file_path_2)

    print(f"Saved 3D figure at: {output_file_path}")
    print(f"Saved 2D figure at: {output_file_path_2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computing the loss landscape with the training trajectory, please model path (.pt) first, weights (.npy) second"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size to evalute the test dataset.""",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="""grid_size^2 amount of points to populate the 2D PCA space to evaluate the loss.""",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="""Subfolder name where the .npy and plots will be saved.""",
    )
    parser.add_argument(
        "--grid_losses",
        type=str,
        default=None,
        help="""Path of the grid losses.""",
    )
    parser.add_argument(
        "--grid_xx",
        type=str,
        default=None,
        help="""Path of the grid xx.""",
    )
    parser.add_argument(
        "--grid_yy",
        type=str,
        default=None,
        help="""Path of the grid yy.""",
    )
    parser.add_argument(
        "--no_grid_save",
        action="store_true",
        help="""Will not save grid xx, grid yy, grid losses and trajectories_losses.""",
    )

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"\nUsing default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(test set)]")
        exit()

    if args.grid_size is not None and (
        args.grid_xx is not None
        or args.grid_yy is not None
        or args.grid_losses is not None
    ):
        print(
            "--grid_size will not be taken into account as the grid_losses, grid_xx and grid_yy are provided."
        )

    if args.grid_size is None:
        args.grid_size = DEFAULT_GRID_SIZE
        print(f"Using default grid_size: {DEFAULT_GRID_SIZE}")
    elif args.grid_size < 1:
        print("Forbidden value !!! grid_size must be > 1")
        exit()

    if len(args.subfolder) > 0:
        print(f"Outputs will be saved to {args.subfolder}/")

    if args.grid_losses is not None:
        if args.grid_xx is None or args.grid_yy is None:
            print("Please provide the three paths.")
            exit()
    elif args.grid_xx is not None:
        if args.grid_losses is None or args.grid_yy is None:
            print("Please provide the three paths.")
            exit()
    elif args.grid_yy is not None:
        if args.grid_losses is None or args.grid_xx is None:
            print("Please provide the three paths.")
            exit()

    if not args.no_grid_save:
        print("Saving trajectories_losses, grid_losses, grid_xx, grid_yy")
    else:
        print("Not saving trajectories_losses, grid_losses, grid_xx, grid_yy")

    main(
        args.batch_size,
        args.subfolder,
        args.grid_size,
        args.grid_losses,
        args.grid_xx,
        args.grid_yy,
        not args.no_grid_save,
    )
