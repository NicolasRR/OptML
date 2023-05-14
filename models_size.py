from common import _get_model, LOSS_FUNC


def print_model_params(dataset_name, light_model):
    model = _get_model(dataset_name, LOSS_FUNC, light_model)
    total_params = sum(p.numel() for p in model.parameters())
    if light_model:
        print(f"Number of parameters of {dataset_name} CNN light: {total_params}\n")
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

