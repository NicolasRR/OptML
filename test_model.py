import argparse
import torch
from sklearn.metrics import classification_report as CR
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import contextlib
import os
from common import (
    _get_model,
    create_testloader,
    create_trainloader,
    compute_accuracy_loss,
    LOSS_FUNC
)

DEFAULT_BATCH_SIZE = 500


def performance(model_path, model, batch_size, classification_report, test=True):
    mode = "test" if test else "train"
    print(f"{mode.capitalize()} Performance")
    loader = (
        create_testloader(model_path, batch_size)
        if test
        else create_trainloader(model_path, batch_size)
    )

    average_accuracy, correct_predictions, average_loss, targets, predictions = compute_accuracy_loss(model, loader, LOSS_FUNC, test_mode=True)

    print(
        f"Average {mode} accuracy: {average_accuracy*100:.2f} % ({correct_predictions}/{len(loader.dataset)})"
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


def compute_training_time_and_pics(model_path, pics, subfolder):
    lines = []
    model_type = None
    if "classic" in model_path:
        model_type = "Classic"
        if len(subfolder) > 0:
            with open(os.path.join(subfolder, "log.log"), "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __main__ - "))
        else:
            with open("log.log", "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __main__ - "))
    elif "async" in model_path:
        model_type = "Asynchronous"
        if len(subfolder) > 0:
            with open(os.path.join(subfolder, "log_async.log"), "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __mp_main__ - "))
        else:
            with open("log_async.log", "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __mp_main__ - "))
    elif "sync" in model_path:
        model_type = "Synchronous"
        if len(subfolder) > 0:
            with open(os.path.join(subfolder, "log_sync.log"), "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __mp_main__ - "))
        else:
            with open("log_sync.log", "r") as log_file:
                # Iterate through each line in the log file
                for line in log_file:
                    lines.append(line.split(" - __mp_main__ - "))
    else:
        print("Unrecognized model path")
        exit()

    for i, line in enumerate(lines):
        timestamp = datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S,%f")
        lines[i][0] = timestamp

    if model_type == "Synchronous" or model_type == "Asynchronous":
        nb_workers = int(
            (lines[0][1].split("with ")[1]).split(" workers")[0]
        )  # extract from first line

    del_list = []
    for i, line in enumerate(lines):
        if "INFO" in line[1]:
            del_list.append(i)

    del_list = sorted(del_list, reverse=True)
    for idx in del_list:
        if idx < len(lines):
            lines.pop(idx)

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
                        int(splited_text[1].split(" ")[3][1:-1].split("/")[0]),
                        int(splited_text[2].split(" ")[2].split("/")[0]),
                    )
                )

            ####### PLOTS #######
            # Create a 1x2 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

            # First subplot (Model Loss vs Time)
            timedeltas = [line[0].total_seconds() for line in model_update_lines]
            losses = [line[1] for line in model_update_lines]

            axs[0].scatter(timedeltas, losses, marker="o")
            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Model Loss")
            axs[0].set_title(
                "Classic SGD evolution of the model loss in function of time"
            )

            # Format x-axis tick labels
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)

            # Second subplot (Cumulative Batch Update Count vs Time)
            batches = [line[2] for line in model_update_lines]

            axs[1].plot(timedeltas, batches)

            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Cumulative Batch Update Count")
            axs[1].set_title("Classic SGD computation speed")

            # Format x-axis tick labels
            axs[1].xaxis.set_major_formatter(formatter)

            if len(subfolder) > 0:
                plt.savefig(
                    os.path.join(subfolder, "model_classic.png"), bbox_inches="tight"
                )
                plt.close(fig)
            else:
                plt.savefig("model_classic.png", bbox_inches="tight")
                plt.close(fig)

        elif model_type == "Synchronous":
            model_loss_lines = []
            worker_update_lines = []
            for i, line in enumerate(lines):
                if i < nb_workers:
                    continue
                if "PS updated model, " in line[1]:
                    # timedetla, model loss
                    model_loss_lines.append((line[0], float(line[1].split("is ")[1])))
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
            # Create a 1x2 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

            # First subplot (Model Loss vs Time)
            timedeltas = [line[0].total_seconds() for line in model_loss_lines]
            losses = [line[1] for line in model_loss_lines]

            axs[0].scatter(timedeltas, losses, marker="o")
            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Model Loss")
            axs[0].set_title(
                "Synchronous SGD evolution of the model loss in function of time"
            )

            # Format x-axis tick labels
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)

            # Second subplot (Cumulative Batch Update Count vs Time)
            worker_cumulative_updates = {}
            for line in worker_update_lines:
                td, worker_id, batch_count, epoch = line
                worker_cumulative_updates.setdefault(worker_id, []).append(
                    (td, batch_count)
                )

            for worker_id, updates in worker_cumulative_updates.items():
                x = [td.total_seconds() for td, _ in updates]
                y = [batch_count for _, batch_count in updates]
                axs[1].plot(x, y, label=f"Worker {worker_id}")

            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Cumulative Batch Update Count")
            axs[1].set_title("Synchronous SGD workers speed comparison")
            axs[1].legend()

            # Format x-axis tick labels
            axs[1].xaxis.set_major_formatter(formatter)

            if len(subfolder) > 0:
                plt.savefig(
                    os.path.join(subfolder, "model_synchronous.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
            else:
                plt.savefig("model_synchronous.png", bbox_inches="tight")
                plt.close(fig)

        elif model_type == "Asynchronous":
            worker_update_lines = []
            for i, line in enumerate(lines):
                if i < nb_workers:
                    continue
                if "PS updated model, worker loss: " in line[1]:
                    splited_text = (line[1].split("PS updated model, worker loss: "))[
                        1
                    ].split(" ")
                    worker_update_lines.append(
                        (
                            line[0],
                            float(splited_text[0]),
                            int(splited_text[1].split("_")[1][:-1]),
                        )
                    )

            ####### PLOTS #######
            # Create a 1x2 grid of subplots
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

            # First subplot (Worker Loss vs Time)
            worker_losses = {}
            for line in worker_update_lines:
                td, worker_loss, worker_id = line
                worker_losses.setdefault(worker_id, []).append((td, worker_loss))

            for worker_id, losses in worker_losses.items():
                x = [td.total_seconds() for td, _ in losses]
                y = [worker_loss for _, worker_loss in losses]
                axs[0].scatter(x, y, label=f"Worker {worker_id}", marker="o")

            axs[0].set_xlabel("Time (MM:SS:sss)")
            axs[0].set_ylabel("Worker Loss")
            axs[0].set_title(
                "Asynchronous SGD evolution of worker loss in function of time"
            )
            axs[0].legend()

            # Format x-axis tick labels
            formatter = FuncFormatter(format_timedelta)
            axs[0].xaxis.set_major_formatter(formatter)

            # Second subplot (Cumulative Batch Update Count vs Time)
            worker_cumulative_updates = {}
            for line in worker_update_lines:
                td, _, worker_id = line
                worker_cumulative_updates.setdefault(worker_id, []).append(td)

            for worker_id, updates in worker_cumulative_updates.items():
                x = [td.total_seconds() for td in updates]
                y = list(range(1, len(updates) + 1))
                axs[1].plot(
                    x,
                    y,
                    label=f"Worker {worker_id}",
                )

            axs[1].set_xlabel("Time (MM:SS:sss)")
            axs[1].set_ylabel("Cumulative Batch Update Count")
            axs[1].set_title("Asynchronous SGD workers speed comparison")
            axs[1].legend()

            # Format x-axis tick labels
            axs[1].xaxis.set_major_formatter(formatter)

            if len(subfolder) > 0:
                plt.savefig(
                    os.path.join(subfolder, "model_asynchronous.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
            else:
                plt.savefig("model_asynchronous.png", bbox_inches="tight")
                plt.close(fig)


def main(model_path, batch_size, classification_report, training_time, pics, subfolder):
    # Load the saved model
    model = _get_model(model_path, LOSS_FUNC)

    model.load_state_dict(torch.load(model_path))

    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        model_filename = os.path.basename(model_path)
        model_basename, _ = os.path.splitext(model_filename)
        output_file_path = os.path.join(subfolder, f"{model_basename}_test_output.txt")

        with open(output_file_path, "w") as output_file:
            with contextlib.redirect_stdout(output_file):
                print(f"Testing performance of {model_path}")
                if training_time:
                    compute_training_time_and_pics(model_path, pics, subfolder)
                performance(
                    model_path, model, batch_size, classification_report, test=False
                )
                performance(
                    model_path, model, batch_size, classification_report, test=True
                )

    print(f"Testing performance of {model_path}")
    if training_time:
        compute_training_time_and_pics(model_path, pics, subfolder)

    performance(model_path, model, batch_size, classification_report, test=False)
    performance(model_path, model, batch_size, classification_report, test=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing models")
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
