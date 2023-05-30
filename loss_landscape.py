import argparse
from sklearn.decomposition import PCA
import plotly.io as pio
import plotly.graph_objs as go
import numpy as np
import torch
import os
from tqdm import tqdm
from common import _get_model, create_testloader, LOSS_FUNC

DEFAULT_GRID_SIZE = 25
DEFAULT_BATCH_SIZE = 1024


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
    weights_path,
    model_path,
    subfolder,
    grid_size,
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

    # Compute the grid border based on the reduced weights
    _min = np.min(reduced_weights) - 1
    _max = np.max(reduced_weights) + 1

    # Compute grid_range_x and grid_range_y
    grid_range_x = np.linspace(_min-1*(_max-_min), _max+1*(_max-_min), grid_size)
    grid_range_y = np.linspace(_min-1*(_max-_min), _max+1*(_max-_min), grid_size)

    # Replace xx, yy with grid_range_x, grid_range_y respectively
    xx, yy = np.meshgrid(grid_range_x, grid_range_y)

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

    grid_losses = np.array(grid_losses)
    grid_losses = np.clip(grid_losses, None, 2*grid_losses.nanmax())
    grid_losses = np.array(grid_losses).reshape(grid_size, grid_size)


    trajectory_loss_reevaluted = []

    # progress_bar2 = tqdm(
    #     total=len(weights_matrix_np),
    #     desc="Computing loss of trajectory weights",
    #     unit="model_weights",
    # )

    # with torch.no_grad():
    #     for weights in weights_matrix_np:
    #         model = set_weights(model, weights)

    #         running_loss = 0.0
    #         for inputs, labels in loader:
    #             outputs = model(inputs)
    #             loss = LOSS_FUNC(outputs, labels)
    #             running_loss += loss.item() * inputs.size(0)

    #         trajectory_loss_reevaluted.append(running_loss / len(loader.dataset))
    #         progress_bar2.update(1)
    #         progress_bar2.set_postfix(trajectory_loss=trajectory_loss_reevaluted[-1])

    # progress_bar2.close()



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

    trajectory = go.Scatter(
        x=reduced_weights[:, 0],
        y=reduced_weights[:, 1],
        mode="markers+lines",
        line=dict(color="red"),
        marker=dict(color=colors, size=sizes),
        name="Training Trajectory",
    )

    # adding a log scale on Z
    layout = go.Layout(
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title=" Loss", zaxis=dict(type='log')),
        coloraxis=dict(
            colorbar=dict(title="Loss magnitude", tickformat=".2e"),  # Format tick labels as scientific notation
            colorscale="Viridis",
            cmin=np.log10(grid_losses.min()),  # Set minimum value on color axis to log10(min(z))
            cmax=np.log10(grid_losses.max()),  # Set maximum value on color axis to log10(max(z))
    ),
    )

    fig = go.Figure(data=[surface], layout=layout)
    
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
    fig2 = go.Figure(data=[go.Contour(x=xx.flatten(), y=yy.flatten(), z=np.log(grid_losses.flatten()), colorscale='Viridis')])

    # Add labels and title
    fig2.update_layout(
        title='Contour Plot with Trajectory Projection',
        xaxis_title='X',
        yaxis_title='Y'
    )
    # Add scatter trace for the trajectory projection

    fig2.add_trace(trajectory)
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
        output_file_path_contour = os.path.join(
            subfolder, f"{model_basename}_loss_landscape_contour.html"
        )
        pio.write_html(fig2, output_file_path_contour)
        np.savetxt(os.path.join(
            subfolder, f"{model_basename}_grid_losses.npy"
        ),np.vstack([xx.flatten(), yy.flatten(), grid_losses.flatten()]))
        np.savetxt(os.path.join(
            subfolder, f"{model_basename}_trajectory_losses.npy"
        ),np.vstack([reduced_weights[:, 0], reduced_weights[:, 1]]))
    else:
        output_file_path = f"{model_basename}_loss_landscape.html"
        pio.write_html(fig, output_file_path)
        output_file_path_contour = f"{model_basename}_loss_landscape_contour.html"
        pio.write_html(fig2, output_file_path_contour)
        np.savetxt(f"{model_basename}_grid_losses.npy"
        ,np.vstack([xx.flatten(), yy.flatten(), grid_losses.flatten()]))
        np.savetxt(f"{model_basename}_trajectory_losses.npy"
        ,np.vstack([reduced_weights[:, 0], reduced_weights[:, 1]]))

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
    )
