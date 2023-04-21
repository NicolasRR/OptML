import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report as CR
from helpers import (
    CNN_MNIST,
    CNN_CIFAR10,
    CNN_CIFAR100,
    create_testloader,
    create_trainloader,
)

DEFAULT_BATCH_SIZE = 500


def performance(model_path, model, batch_size, classification_report, test=True):
    if test:
        print("Test Performance")
        test_loader = create_testloader(model_path, batch_size)
        mode = "test"
    else:
        print("Train Performance")
        test_loader = create_trainloader(model_path, batch_size)
        mode = "train"

    # Evaluate the model on the test dataset
    test_loss = 0
    correct = 0
    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            targets.extend(target.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(
        f"Average {mode} accuracy: {accuracy*100:.2f} % ({correct}/{len(test_loader.dataset)})"
    )
    print(f"Average {mode} loss: {test_loss:.4f}")

    if classification_report:
        report = CR(targets, predictions, zero_division=0)
        print(f"\nClassification {mode} report:")
        print(report)


def main(model_path, batch_size, classification_report):
    print(f"Testing performance of {model_path}")
    # Load the saved model
    model = 0
    if "mnist" in model_path:
        print("Loading MNIST CNN\n")
        model = CNN_MNIST()
    elif "cifar100" in model_path:
        print("Loading CIFAR100 CNN\n")
        model = CNN_CIFAR100()
    elif "cifar10" in model_path:
        print("Loading CIFAR10 CNN\n")
        model = CNN_CIFAR10()
    else:
        print("dataset not supported\n")
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

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    main(args.model_path.name, args.batch_size, args.classification_report)
