import argparse
import torch
from sklearn.metrics import classification_report as CR
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import contextlib
import os
import numpy as np
import sys
from common import (
    _get_model,
    create_testloader,
    create_trainloader,
    compute_accuracy_loss,
    LOSS_FUNC,
)

DEFAULT_BATCH_SIZE = 500


class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


@contextlib.contextmanager
def redirect_stdout_to_file(file):
    sys.stdout = Tee(file)
    try:
        yield
    finally:
        sys.stdout = sys.__stdout__


def performance(model_path, model, batch_size, classification_report, test=True):
    mode = "test" if test else "train"
    print(f"{mode.capitalize()} Performance")
    loader = (
        create_testloader(model_path, batch_size)
        if test
        else create_trainloader(model_path, batch_size)
    )

    (
        average_accuracy,
        correct_predictions,
        total_predictions,
        average_loss,
        targets,
        predictions,
    ) = compute_accuracy_loss(model, loader, LOSS_FUNC, test_mode=True)

    print(
        f"Average {mode} accuracy: {average_accuracy*100:.2f} % ({correct_predictions}/{total_predictions})"
    )
    print(f"Average {mode} loss: {average_loss:.4f}")

    if classification_report:
        report = CR(targets, predictions, zero_division=0)
        print(f"\nClassification {mode} report:")
        print(report)


def format_timedelta(x, _):
    td = timedelta(seconds=x)
    minutes, remainder = divmod(td.seconds, 60)
    seconds = remainder
    milliseconds = td.microseconds // 1000
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"


def save_fig(fig, subfolder, model_path, validation=False):
    model_filename = os.path.basename(model_path)
    model_basename, _ = os.path.splitext(model_filename)

    if validation == False:
        save_name = f"{model_basename}_training_plots.png"
        if len(subfolder) > 0:
            output_file_path = os.path.join(subfolder, save_name)
            plt.savefig(output_file_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved fig at {output_file_path}")
        else:
            plt.savefig(save_name, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved fig at {save_name}")
    else:
        save_name = f"{model_basename}_validation_plots.png"
        if len(subfolder) > 0:
            output_file_path = os.path.join(subfolder, save_name)
            plt.savefig(output_file_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved fig at {output_file_path}")
        else:
            plt.savefig(save_name, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved fig at {save_name}")


def compute_training_time_and_pics(model_path, pics, subfolder):
    lines = []
    model_type = None
    log_file_name = None
    if "classic" in model_path:
        model_type = "Classic"
        log_file_name = "log.log"
    elif "async" in model_path:
        model_type = "Asynchronous"
        log_file_name = "log_async.log"
    elif "sync" in model_path:
        model_type = "Synchronous"
        log_file_name = "log_sync.log"
    else:
        print("Unrecognized model path")
        exit()

    if len(subfolder) > 0:
        with open(os.path.join(subfolder, log_file_name), "r") as log_file:
            # Iterate through each line in the log file
            for line in log_file:
                lines.append(line.split(" - common - "))
    else:
        with open(log_file_name, "r") as log_file:
            for line in log_file:
                lines.append(line.split(" - common - "))

    for i, line in enumerate(lines):
        timestamp = datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S,%f")
        lines[i][0] = timestamp

    nb_workers = None
    if model_type == "Synchronous" or model_type == "Asynchronous":
        nb_workers = int((lines[0][1].split("with ")[1]).split(" workers")[0])

    lines = [line for line in lines if "INFO" not in line[1]]

    val_lines = None
    val_lines = [line for line in lines if "Train loss:" in line[1]]
    lines = [line for line in lines if "Train loss:" not in line[1]]

    if model_type == "Synchronous" or model_type == "Asynchronous":
        lines = lines[nb_workers:]

    start_time = lines[0][0]
    end_time = lines[-1][0]
    training_time = end_time - start_time

    minutes, remainder = divmod(training_time.seconds, 60)
    seconds = remainder
    milliseconds = training_time.microseconds // 1000
    formatted_training_time = f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"
    print(f"{model_type} training time:", formatted_training_time)  # MM:SS:sss

    if pics:
        for i, line in enumerate(lines):
            lines[i][0] = line[0] - start_time
            lines[i][1] = line[1].split("DEBUG - ")[1].strip()

        if model_type == "Classic":
            model_update_lines = []
            for line in lines:
                splited_text = line[1].split(",")
                model_update_lines.append(
                    (
                        line[0],
                        float(splited_text[0].split(" ")[1]),
                        float(splited_text[1].split(" ")[-1]),
                        int(splited_text[2].split(" ")[3][1:-1].split("/")[0]),
                        int(splited_text[3].split(" ")[-1].split("/")[0]),
                    )
                )

            ####### PLOTS #######
            # Create a 1x3 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26, 5))
            # First subplot (Model Loss vs Time)
            timedeltas = [line[0].total_seconds() for line in model_update_lines]
            losses = [line[1] for line in model_update_lines]
            axs[0].plot(timedeltas, losses)
            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Model Loss")
            axs[0].set_title(
                "Classic SGD evolution of the model loss in function of time"
            )
            # Format x-axis tick labels
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)
            # Second subplot (Weights L2 norm vs Time)
            weights_norm = [line[2] for line in model_update_lines]
            axs[1].plot(timedeltas, weights_norm)
            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Weights L2 norm")
            axs[1].set_title(
                "Classic SGD evolution of the model weights L2 norm in function of time"
            )
            axs[1].xaxis.set_major_formatter(formatter)
            # Thirst subplot (Cumulative Batch Update Count vs Time)
            batches = [line[3] for line in model_update_lines]
            axs[2].plot(timedeltas, batches)
            axs[2].set_xlabel("Time (MM:SS:sss)")
            axs[2].set_ylabel("Cumulative Batch Update Count")
            axs[2].set_title("Classic SGD computation speed")
            axs[2].xaxis.set_major_formatter(formatter)

            save_fig(fig, subfolder, model_path)

        elif model_type == "Synchronous":
            model_loss_lines = []
            worker_update_lines = []
            for i, line in enumerate(lines):
                if "PS updated model, " in line[1]:
                    # timedetla, model loss
                    model_loss_lines.append(
                        (
                            line[0],
                            float(line[1].split(",")[1].split(" ")[-1]),
                            float(line[1].split(",")[2].split(" ")[-1]),
                        )
                    )
                elif "PS got " in line[1]:
                    splited_text = (line[1].split("from ")[1]).split(" ")
                    # timedelta, worker id, batch count, epoch
                    worker_update_lines.append(
                        (
                            line[0],
                            int(splited_text[0].split("_")[1][:-1]),
                            int(splited_text[2].split("/")[0][1:]),
                            int(splited_text[-1].split("/")[0]),
                        )
                    )

            ####### PLOTS #######
            # Create a 1x3 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26, 5))
            # First subplot (Model Loss vs Time)
            timedeltas = [line[0].total_seconds() for line in model_loss_lines]
            losses = [line[1] for line in model_loss_lines]
            axs[0].plot(timedeltas, losses)
            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Model Loss")
            axs[0].set_title(
                "Synchronous SGD evolution of the model loss in function of time"
            )
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)
            # Second subplot (Weights L2 norm vs Time)
            weights_norm = [line[2] for line in model_loss_lines]
            axs[1].plot(timedeltas, weights_norm)
            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Weights L2 norm")
            axs[1].set_title(
                "Synchronous SGD evolution of the model weights L2 norm in function of time"
            )
            formatter = FuncFormatter(format_timedelta)
            axs[1].xaxis.set_major_formatter(formatter)
            # Third subplot (Cumulative Batch Update Count vs Time)
            worker_cumulative_updates = {}
            for line in worker_update_lines:
                td, worker_id, batch_count, epoch = line
                worker_cumulative_updates.setdefault(worker_id, []).append(
                    (td, batch_count)
                )
            for worker_id, updates in worker_cumulative_updates.items():
                x = [td.total_seconds() for td, _ in updates]
                y = [batch_count for _, batch_count in updates]
                axs[2].plot(x, y, label=f"Worker {worker_id}")
            axs[2].set_xlabel("Time (MM:SS:sss)")
            axs[2].set_ylabel("Cumulative Batch Update Count")
            axs[2].set_title("Synchronous SGD workers speed comparison")
            axs[2].legend()
            axs[2].xaxis.set_major_formatter(formatter)

            save_fig(fig, subfolder, model_path)

        elif model_type == "Asynchronous":
            worker_update_lines = []
            for i, line in enumerate(lines):
                if "PS updated model, worker loss: " in line[1]:
                    splited_text = (line[1].split("PS updated model, worker loss: "))[
                        1
                    ].split(" ")
                    worker_update_lines.append(
                        (
                            line[0],
                            float(splited_text[0]),
                            float(splited_text[-1]),
                            int(splited_text[1].split("_")[1].split(")")[0]),
                        )
                    )

            ####### PLOTS #######
            # Create a 1x3 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(26, 5))
            # First subplot (Worker Loss vs Time)
            worker_losses = {}
            for line in worker_update_lines:
                td, worker_loss, _, worker_id = line
                worker_losses.setdefault(worker_id, []).append((td, worker_loss))
            for worker_id, losses in worker_losses.items():
                x = [td.total_seconds() for td, _ in losses]
                y = [worker_loss for _, worker_loss in losses]
                axs[0].plot(x, y, label=f"Worker {worker_id}")
            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Worker Loss")
            axs[0].set_title(
                "Asynchronous SGD evolution of worker loss in function of time"
            )
            axs[0].legend()
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)
            # Second subplot (Weigh L2 norm vs Time)
            weights_norms = {}
            for line in worker_update_lines:
                td, _, worker_w_norm, worker_id = line
                weights_norms.setdefault(worker_id, []).append((td, worker_w_norm))
            for worker_id, weight_norm in weights_norms.items():
                x = [td.total_seconds() for td, _ in weight_norm]
                y = [w_norm for _, w_norm in weight_norm]
                axs[1].plot(x, y, label=f"Worker {worker_id}")
            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Weights L2 norm")
            axs[1].set_title(
                "Asynchronous SGD evolution of the weights L2 norm in function of time"
            )
            axs[1].legend()
            formatter = FuncFormatter(format_timedelta)
            axs[1].xaxis.set_major_formatter(formatter)
            # Third subplot (Cumulative Batch Update Count vs Time)
            worker_cumulative_updates = {}
            for line in worker_update_lines:
                td, _, _, worker_id = line
                worker_cumulative_updates.setdefault(worker_id, []).append(td)
            for worker_id, updates in worker_cumulative_updates.items():
                x = [td.total_seconds() for td in updates]
                y = list(range(1, len(updates) + 1))
                axs[2].plot(x, y, label=f"Worker {worker_id}")
            axs[2].set_xlabel("Time (MM:SS:sss)")
            axs[2].set_ylabel("Cumulative Batch Update Count")
            axs[2].set_title("Asynchronous SGD workers speed comparison")
            axs[2].legend()
            # Format x-axis tick labels
            axs[2].xaxis.set_major_formatter(formatter)

            save_fig(fig, subfolder, model_path)

        if val_lines is not None:
            if len(val_lines) > 0:
                time_offset = val_lines[0][0]
                for idx, v_line in enumerate(val_lines):
                    splited_text = v_line[1].split(", ")
                    val_lines[idx] = (
                        v_line[0] - time_offset,
                        float(splited_text[0].split("Train loss: ")[-1]),  # train loss
                        float(
                            splited_text[1].split("train accuracy: ")[1].split(" ")[0]
                        ),  # train accuracy
                        float(splited_text[2].split("val loss: ")[-1]),  # val loss
                        float(
                            splited_text[3].split("val accuracy: ")[1].split(" ")[0]
                        ),  # val accuracy
                    )
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
                timedeltas = [line[0].total_seconds() for line in val_lines]
                tr_losses = [line[1] for line in val_lines]
                tr_acc = [line[2] for line in val_lines]
                val_losses = [line[3] for line in val_lines]
                val_acc = [line[4] for line in val_lines]

                epochs = list(range(1, len(val_lines) + 1))
                epoch_opt = np.argmin(val_losses) + 1
                print(f"The early stopping epoch is epoch {epoch_opt}")

                # First subplot (Model Loss vs Epoch)
                axs[0].plot(epochs, tr_losses, marker="x", label="Training")
                axs[0].plot(epochs, val_losses, marker="x", label="Validation")
                axs[0].axvline(
                    epoch_opt, color="k", linestyle="--", label="Early Stopping Point"
                )
                if model_type == "Asynchronous":
                    axs[0].set_xlabel("Pseudo Epochs")
                else:
                    axs[0].set_xlabel("Epochs")
                axs[0].set_ylabel("Loss")
                axs[0].set_title(
                    f"{model_type} SGD evolution of the train and validation loss in function of epoch"
                )
                axs[0].legend()
                # Second subplot (Weights L2 norm vs Epoch)
                axs[1].plot(epochs, tr_acc, marker="x", label="Training")
                axs[1].plot(epochs, val_acc, marker="x", label="Validation")
                axs[1].axvline(
                    epoch_opt, color="k", linestyle="--", label="Early Stopping Point"
                )
                if model_type == "Asynchronous":
                    axs[1].set_xlabel("Pseudo Epochs")
                else:
                    axs[1].set_xlabel("Epochs")
                axs[1].set_ylabel("Accuracy")
                axs[1].set_title(
                    f"{model_type} SGD evolution of the train and validation loss in function of epoch"
                )

                axs[1].legend()

                save_fig(fig, subfolder, model_path, validation=True)


def main(
    model_path,
    batch_size,
    classification_report,
    training_time,
    pics,
    subfolder,
):
    # Load the saved model
    if "alt_model" in model_path:
        model = _get_model(model_path, LOSS_FUNC, alt_model=True)
    else:
        model = _get_model(model_path, LOSS_FUNC, alt_model=False)

    model.load_state_dict(torch.load(model_path))

    model_filename = os.path.basename(model_path)
    model_basename, _ = os.path.splitext(model_filename)

    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
            output_file_path = os.path.join(
                subfolder, f"{model_basename}_test_output.txt"
            )
    else:
        output_file_path = f"{model_basename}_test_output.txt"

    print(f"Saving to outputs to {output_file_path}")

    with open(output_file_path, "w") as output_file:
        with redirect_stdout_to_file(output_file):
            print(f"Testing performance of {model_path}")
            if training_time:
                compute_training_time_and_pics(model_path, pics, subfolder)
            performance(
                model_path, model, batch_size, classification_report, test=False
            )
            performance(model_path, model, batch_size, classification_report, test=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computing train and test performance, reading .log and generating images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""",
    )
    parser.add_argument(
        "model_path", type=argparse.FileType("r"), help="""Path of the trained model."""
    )
    parser.add_argument(
        "--classification_report",
        action="store_true",
        help="""If set, prints a classification report (labels performance).""",
    )
    parser.add_argument(
        "--training_time",
        action="store_true",
        help="""If set, will read the associated log file to compute the training time.""",
    )
    parser.add_argument(
        "--pics",
        action="store_true",
        help="""If set, will compute and save plots from the .log file.""",
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
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    if args.pics and not args.training_time:
        print("Pictures mode only with --training_time")
        exit()

    if len(args.subfolder) > 0:
        print(f"Outputs will be saved to {args.subfolder}/")

    main(
        args.model_path.name,
        args.batch_size,
        args.classification_report,
        args.training_time,
        args.pics,
        args.subfolder,
    )
