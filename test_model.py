import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report as CR
from datetime import datetime
from helpers import (
    CNN_MNIST,
    CNN_CIFAR10,
    CNN_CIFAR100,
    create_testloader,
    create_trainloader,
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

    test_loss = 0
    correct = 0
    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            targets.extend(target.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())

    test_loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    print(
        f"Average {mode} accuracy: {accuracy*100:.2f} % ({correct}/{len(loader.dataset)})"
    )
    print(f"Average {mode} loss: {test_loss:.4f}")

    if classification_report:
        report = CR(targets, predictions, zero_division=0)
        print(f"\nClassification {mode} report:")
        print(report)


def compute_training_time(model_path):
    lines = []
    model_type = None
    if "classic" in model_path:
        model_type = "Classic"
        with open("log.log", "r") as log_file:
            # Iterate through each line in the log file
            for line in log_file:
                lines.append(line.split(" - __main__ - "))
    elif "async" in model_path:
        model_type = "Asynchronous"
        with open("log_async.log", "r") as log_file:
            # Iterate through each line in the log file
            for line in log_file:
                lines.append(line.split(" - __mp_main__ - "))
    elif "sync" in model_path:
        model_type = "Synchronous"
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


def main(model_path, batch_size, classification_report, training_time):
    print(f"Testing performance of {model_path}")

    if training_time:
        compute_training_time(model_path)

    # Load the saved model
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
        help="If set, prints a classification report (labels performance)",
    )
    parser.add_argument(
        "--training_time",
        action="store_true",
        help="If set, will read the associated log file to compute the training time",
    )

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"\nUsing default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    main(
        args.model_path.name,
        args.batch_size,
        args.classification_report,
        args.training_time,
    )
