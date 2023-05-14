from common import _get_model, LOSS_FUNC


def print_model_params(dataset_name, alt_model):
    model = _get_model(dataset_name, LOSS_FUNC, alt_model)
    total_params = sum(p.numel() for p in model.parameters())
    if alt_model:
        print(f"Number of parameters of {dataset_name} CNN (alternative): {total_params}\n")
    else:
        print(f"Number of parameters of {dataset_name}: {total_params}\n")

if __name__ == "__main__":
    print_model_params("mnist", False)
    print_model_params("mnist", True)
    print_model_params("fashion_mnist", False)
    print_model_params("fashion_mnist", True)
    print_model_params("cifar10", False)
    print_model_params("cifar10", True)
    print_model_params("cifar100", False)
    print_model_params("cifar100", True)

