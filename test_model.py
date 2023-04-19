import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from helpers import CNN_MNIST, CNN_CIFAR, create_testloader

DEFAULT_BATCH_SIZE = 500


def main(model_path, batch_size):
    # Load the saved model
    model = 0
    if "mnist" in model_path:
        print("Loading MNIST CNN")
        model = CNN_MNIST()
    elif "cifar" in model_path:
        print("Loading CIFAR CNN")
        model = CNN_CIFAR()

    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader, nb_labels = create_testloader(model_path, batch_size)

    # Evaluate the model on the test dataset
    test_loss = 0
    correct = 0
    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data, target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            targets.extend(target.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print(
        f"Average accuracy: {accuracy*100:.2f} % ({correct}/{len(test_loader.dataset)})"
    )
    print(f"Average loss: {test_loss:.4f}")

    report = classification_report(targets, predictions)
    print("\nClassification report:")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing models")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""",
    )
    parser.add_argument(
        "remainder", nargs=argparse.REMAINDER, help="""Path of the trained model."""
    )

    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")
    elif args.batch_size < 1:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    model_path = None
    for arg in args.remainder:
        if arg.startswith("--"):
            continue
        else:
            model_path = arg
            break

    if model_path is None:
        print("Missing model path !!!")
        exit()

    main(model_path, args.batch_size)