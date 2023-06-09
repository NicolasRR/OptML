import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
import numpy as np
import logging
import logging.handlers
import queue
import os
from multiprocessing import Manager
import threading
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import time
from sklearn.metrics import classification_report as CR


DEFAULT_DATASET = "mnist"
DEFAULT_WORLD_SIZE = 3
DEFAULT_TRAIN_SPLIT = 1
DEFAULT_LR = 1e-2
DEFAULT_MOMENTUM = 0.0
DEFAULT_EPOCHS = 1
DEFAULT_SEED = 614310
# LOSS_FUNC = nn.functional.cross_entropy
LOSS_FUNC = nn.functional.nll_loss  # add softmax layer if nll
EXPO_DECAY = 0.9  # for exponential learning rate scheduler
DEFAULT_BATCH_SIZE = 32  # 1 == SGD, >1 MINI BATCH SGD
DEFAULT_DELAY_INTENSITY = "small"
DEFAULT_DELAY_TYPE = "gaussian"
DELAY_VALUES = {
    "small": (0.03, 0.005),  # 10 ms, 1 ms std
    "medium": (0.06, 0.005),  # 20 ms, 1 ms std
    "long": (0.09, 0.005),  # 30 ms, 1 ms std
}
DELAY_WORKER_1_FACTOR = 2
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_ALTERNATE_MODELS = False


#################################### Start and Run ####################################
def start(args, mode, run_parameter_server):
    if mode == "sync":
        base_name = get_base_name(
            mode,
            args.dataset,
            args.world_size,
            args.train_split,
            args.lr,
            args.momentum,
            args.batch_size,
            args.epochs,
            args.val,
            args.alr,
            args.lrs,
            args.saves_per_epoch,
            alt_model=args.alt_model,
            split_dataset=args.split_dataset,
            split_labels=args.split_labels,
            delay=args.delay,
            slow_worker_1=args.slow_worker_1,
            delay_intensity=args.delay_intensity,
            delay_type=args.delay_type,
        )
    if mode == "async":
        base_name = get_base_name(
            mode,
            args.dataset,
            args.world_size,
            args.train_split,
            args.lr,
            args.momentum,
            args.batch_size,
            args.epochs,
            args.val,
            args.alr,
            args.lrs,
            args.saves_per_epoch,
            alt_model=args.alt_model,
            split_dataset=args.split_dataset,
            split_labels=args.split_labels,
            split_labels_unscaled=args.split_labels_unscaled,
            delay=args.delay,
            slow_worker_1=args.slow_worker_1,
            delay_intensity=args.delay_intensity,
            delay_type=args.delay_type,
        )

    log_name = f"{base_name}_log.log"

    with Manager() as manager:
        log_queue = manager.Queue()
        log_writer_thread = threading.Thread(
            target=log_writer, args=(log_queue, args.subfolder, log_name)
        )
        log_writer_thread.start()
        mp.spawn(
            run,
            args=(
                mode,
                log_queue,
                args.dataset,
                args.split_dataset,
                args.split_labels,
                args.split_labels_unscaled
                if hasattr(args, "split_labels_unscaled")
                else None,
                args.world_size,
                args.lr,
                args.momentum,
                args.train_split,
                args.batch_size,
                args.epochs,
                args.worker_accuracy,
                args.model_accuracy,
                not args.no_save_model,
                args.subfolder,
                args.saves_per_epoch,
                args.alr,
                args.lrs,
                args.delay,
                args.delay_intensity,
                args.delay_type,
                args.slow_worker_1,
                args.val,
                args.alt_model,
                run_parameter_server,
            ),
            nprocs=args.world_size,
            join=True,
        )
        log_queue.put(None)
        log_writer_thread.join()


def run(
    rank,
    mode,
    log_queue,
    dataset_name,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    world_size,
    learning_rate,
    momentum,
    train_split,
    batch_size,
    epochs,
    worker_accuracy,
    model_accuracy,
    save_model,
    subfolder,
    saves_per_epoch,
    use_alr,
    lrs,
    delay,
    delay_intensity,
    delay_type,
    slow_woker_1,
    val,
    alt_model,
    run_parameter_server,
):
    logger = setup_logger(log_queue)
    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=world_size, rpc_timeout=0
    )
    if rank != 0:
        rpc.init_rpc(
            f"Worker_{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
    else:
        rpc.init_rpc(
            "Parameter_Server",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        if mode == "sync":
            run_parameter_server(
                [f"Worker_{r}" for r in range(1, world_size)],
                logger,
                dataset_name,
                split_dataset,
                split_labels,
                learning_rate,
                momentum,
                train_split,
                batch_size,
                epochs,
                worker_accuracy,
                model_accuracy,
                save_model,
                subfolder,
                use_alr,
                saves_per_epoch,
                lrs,
                delay,
                delay_intensity,
                delay_type,
                slow_woker_1,
                val,
                alt_model,
            )
        elif mode == "async":
            run_parameter_server(
                [f"Worker_{r}" for r in range(1, world_size)],
                logger,
                dataset_name,
                split_dataset,
                split_labels,
                split_labels_unscaled,
                learning_rate,
                momentum,
                train_split,
                batch_size,
                epochs,
                worker_accuracy,
                model_accuracy,
                save_model,
                subfolder,
                use_alr,
                saves_per_epoch,
                lrs,
                delay,
                delay_intensity,
                delay_type,
                slow_woker_1,
                val,
                alt_model,
            )
    rpc.shutdown()


#################################### Parser ####################################
def check_args(args, mode):
    if args.dataset is None:
        args.dataset = DEFAULT_DATASET
        print(f"Using default dataset: {DEFAULT_DATASET}")

    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
        print(f"Using default batch_size: {DEFAULT_BATCH_SIZE}")

    if mode is not None:
        if args.world_size is None:
            args.world_size = DEFAULT_WORLD_SIZE
            print(f"Using default world_size value: {DEFAULT_WORLD_SIZE}")
        elif args.world_size < 2:
            print(
                "Forbidden value !!! world_size must be >= 2 (1 Parameter Server and 1 Worker)"
            )
            exit()

        if args.split_labels and args.split_dataset:
            print("Please use --split_labels without --split_dataset")
            exit()

        if args.delay_intensity is not None and (
            not args.delay and not args.slow_worker_1
        ):
            print("Please use --delay_intensity with --delay or --slow_worker_1")
            exit()

        if args.delay_type is not None and (not args.delay and not args.slow_worker_1):
            print("Please use --delay_type with --delay or --slow_worker_1")
            exit()

        if args.delay:
            print("Adding delay to all workers")

        if args.slow_worker_1:
            print("Slowing down worker 1 with larger delay")

        if (args.delay or args.slow_worker_1) and args.delay_intensity is None:
            args.delay_intensity = DEFAULT_DELAY_INTENSITY
            print(f"Using the default delay intensity: {args.delay_intensity}")

        if (args.delay or args.slow_worker_1) and args.delay_type is None:
            args.delay_type = DEFAULT_DELAY_TYPE
            print(f"Using the default delay type: {args.delay_type}")

        if mode == "async":
            if args.split_labels_unscaled and args.split_dataset:
                print("Please use --split_labels_unscaled without --split_dataset")
                exit()

            if args.split_labels and args.split_labels_unscaled:
                print(
                    "Please do not use --split_labels and --split_labels_unscaled together"
                )
                exit()

        if args.dataset == "mnist":
            valid_world_sizes = {3, 6, 11}
        elif args.dataset == "fashion_mnist":
            valid_world_sizes = {3, 5, 6, 9, 11, 21, 41}
        elif args.dataset == "cifar10":
            valid_world_sizes = {3, 6, 11}
        elif args.dataset == "cifar100":
            valid_world_sizes = {3, 5, 6, 11, 21, 26, 51, 101}
        if mode == "sync":
            if args.split_labels:
                if args.world_size not in valid_world_sizes:
                    print(
                        f"Please use --split_labels with --world_size {valid_world_sizes}"
                    )
                    exit()
        elif mode == "async":
            if args.split_labels or args.split_labels_unscaled:
                if args.world_size not in valid_world_sizes:
                    if args.split_labels:
                        print(
                            f"Please use --split_labels with --world_size {valid_world_sizes}"
                        )
                        exit()
                    else:
                        print(
                            f"Please use --split_labels_unscaled with --world_size {valid_world_sizes}"
                        )
                        exit()

    if args.train_split is None:
        args.train_split = DEFAULT_TRAIN_SPLIT
        print(f"Using default train_split value: {DEFAULT_TRAIN_SPLIT}")
    elif args.train_split > 1 or args.train_split <= 0:
        print("Forbidden value !!! train_split must be between (0,1]")
        exit()

    if args.lr is None:
        args.lr = DEFAULT_LR
        print(f"Using default lr: {DEFAULT_LR}")
    elif args.lr <= 0:
        print("Forbidden value !!! lr must be between (0,+inf)")
        exit()

    if args.momentum is None:
        args.momentum = DEFAULT_MOMENTUM
        print(f"Using default momentum: {DEFAULT_MOMENTUM}")
    elif args.momentum < 0:
        print("Forbidden value !!! momentum must be between [0,+inf)")
        exit()

    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS
        print(f"Using default epochs: {DEFAULT_EPOCHS}")
    elif args.epochs < 1:
        print("Forbidden value !!! epochs must be between [1,+inf)")
        exit()

    if args.saves_per_epoch is not None:
        if args.saves_per_epoch < 1:
            print("Forbidden value !!! saves_per_epoch must be > 1")
            exit()
        else:
            print(f"Saving model weights {args.saves_per_epoch} times during one epoch")

    if args.seed:
        torch.manual_seed(DEFAULT_SEED)
        np.random.seed(DEFAULT_SEED)

    if len(args.subfolder) > 0:
        print(f"Saving model and log file to {args.subfolder}")

    if args.alr:
        print("Using Adam as optimizer instead of SGD")

    if args.lrs is not None:
        print(f"Using learning rate scheduler: {args.lrs}")

    if mode == "sync" or mode == "async":
        if args.val and args.split_dataset:
            print("Please do not use --args.val and --args.split_dataset together")
            exit()
        elif args.val and args.split_labels:
            print("Please do not use --args.val and --args.split_labels together")
            exit()

        if mode == "async":
            if args.val and args.split_labels_unscaled:
                print(
                    "Please do not use --args.val and --args.split_labels_unscaled together"
                )
                exit()

        elif args.val:
            print("Using validation to analyze regularization.")

    return args


def read_parser(parser, mode=None):
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        default=None,
        help="Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100.",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=None,
        help="""Fraction of the training dataset to be used for training (0,1].""",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="""Learning rate of SGD  (0,+inf)."""
    )
    parser.add_argument(
        "--momentum", type=float, default=None, help="""Momentum of SGD  [0,+inf)."""
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="""Batch size of Mini batch SGD [1,len(train set)].""",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="""Number of epochs for training [1,+inf).""",
    )
    parser.add_argument(
        "--model_accuracy",
        action="store_true",
        help="""If set, will compute the train accuracy of the global model after training.""",
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="""If set, the trained model will not be saved.""",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="""If set, it will set seeds on torch, numpy and random for reproducibility purposes.""",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="",
        help="""Subfolder where the model and log.log will be saved.""",
    )
    parser.add_argument(
        "--saves_per_epoch",
        type=int,
        default=None,
        help="""Number of times the model weights will be saved during one epoch.""",
    )
    parser.add_argument(
        "--alr",
        action="store_true",
        help="""If set, use adaptive learning rate (Adam optimizer) instead of SGD optimizer.""",
    )
    parser.add_argument(
        "--lrs",
        type=str,
        choices=["exponential", "cosine_annealing"],
        default=None,
        help="""Choose a learning rate scheduler: exponential or cosine_annealing.""",
    )
    parser.add_argument(
        "--alt_model",
        action="store_true",
        help="""Will train using alternate CNN models instead of LeNet5 or ResNet18.""",
    )
    if mode is None:
        parser.add_argument(
            "--val",
            action="store_true",
            help="""If set, will create a validation loader and compute the loss and accuracy of train and val at the end of each epoch.""",
        )
    if mode == "sync":
        parser.add_argument(
            "--val",
            action="store_true",
            help="""If set, will create a validation loader and compute the loss and accuracy of train and val at the end of each epoch. Please use --val without --split_dataset or --split_labels.""",
        )
    if mode == "async":
        parser.add_argument(
            "--val",
            action="store_true",
            help="""If set, will create a validation loader and compute the loss and accuracy of train and val at the end of each epoch. Please use --val without --split_dataset or --split_labels or --split_labels_unscaled.""",
        )
    if mode is not None:
        parser.add_argument(
            "--master_port",
            type=str,
            default="29500",
            help="""Port that master is listening on, will default to 29500 if not
            provided. Master must be able to accept network traffic on the host and port.""",
        )
        parser.add_argument(
            "--master_addr",
            type=str,
            default="localhost",
            help="""Address of master, will default to localhost if not provided.
            Master must be able to accept network traffic on the address + port.""",
        )
        parser.add_argument(
            "--world_size",
            type=int,
            default=None,
            help="""Total number of participating processes. Should be the sum of
            master node and all training nodes [2,+inf].""",
        )
        parser.add_argument(
            "--worker_accuracy",
            action="store_true",
            help="""If set, will compute the train accuracy of each worker after training (useful when --split_dataset).""",
        )
        parser.add_argument(
            "--delay",
            action="store_true",
            help="""Add a delay to all workers at each mini-batch update.""",
        )
        parser.add_argument(
            "--slow_worker_1",
            action="store_true",
            help="""Add a longer delay only to worker 1 at each mini-batch update.""",
        )
        parser.add_argument(
            "--delay_intensity",
            type=str,
            choices=["small", "medium", "long"],
            default=None,
            help="""Choose the delay intensity: small 10ms, medium 20ms, long 30ms.""",
        )
        parser.add_argument(
            "--delay_type",
            type=str,
            choices=["constant", "gaussian"],
            default=None,
            help="""Choose the delay type: constant or gaussian.""",
        )
        if mode is not None:
            parser.add_argument(
                "--split_dataset",
                action="store_true",
                help="""After applying train_split, each worker will train on a unique distinct dataset (samples will not be 
                shared between workers).""",
            )
        if mode == "sync":
            parser.add_argument(
                "--split_labels",
                action="store_true",
                help="""If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. 
                Workers will not share samples and the labels are randomly assigned.
                Don't use with --split_dataset. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
            )
        elif mode == "async":
            parser.add_argument(
                "--split_labels",
                action="store_true",
                help="""If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. 
                Workers will not share samples and the labels are randomly assigned.
                Don't use with --split_dataset or --split_labels_unscaled. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
            )
            parser.add_argument(
                "--split_labels_unscaled",
                action="store_true",
                help="""If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. 
                Workers will not share samples and the labels are randomly assigned. Note, the training length will be the DIFFERENT for all workers, based on the number of samples each class has.
                Don't use --split_dataset or split_labels. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded.""",
            )

    args = parser.parse_args()
    if mode is not None:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port

    return check_args(args, mode)


#################################### Main utility functions ####################################
def _get_model(dataset_name, loss_func, alt_model=DEFAULT_ALTERNATE_MODELS):
    if alt_model == False:
        if "mnist" in dataset_name:
            print("Created MNIST/FASHION_MNIST CNN")
            return CNN_MNIST(loss_func=loss_func)  # global model
        elif "cifar100" in dataset_name:
            print("Created CIFAR100 CNN")
            return CNN_CIFAR100(loss_func=loss_func)
        elif "cifar10" in dataset_name:
            print("Created CIFAR10 CNN")
            return CNN_CIFAR10(loss_func=loss_func)
        else:
            print("Unknown dataset, cannot create CNN")
            exit()
    else:
        if "mnist" in dataset_name:
            print("Created MNIST/FASHION_MNIST CNN (alternative)")
            return CNN_MNIST_alt(loss_func=loss_func)  # global model
        elif "cifar100" in dataset_name:
            print("Created CIFAR100 CNN (alternative)")
            return CNN_CIFAR100_alt(loss_func=loss_func)
        elif "cifar10" in dataset_name:
            print("Created CIFAR10 CNN (alternative)")
            return CNN_CIFAR10_alt(loss_func=loss_func)
        else:
            print("Unknown dataset, cannot create CNN (alternative)")
            exit()


def get_optimizer(model, learning_rate, momentum, use_alr):
    if use_alr:
        if momentum > 1:
            return optim.Adam(model.parameters(), lr=learning_rate)
        else:
            return optim.Adam(
                model.parameters(), lr=learning_rate, betas=(max(momentum, 0.99), 0.999)
            )  # weight decay if weights too large
    else:
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def get_scheduler(lrs, optimizer, len_trainloader, epochs, gamma=EXPO_DECAY):
    if lrs is not None:
        if lrs == "exponential":  # more suitable for async
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )  # initial_learning_rate * gamma^epoch
        elif lrs == "cosine_annealing":  # more suitable for sync
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=len_trainloader * epochs
            )
        else:
            return None


def compute_accuracy_loss(
    model,
    loader,
    loss_func,
    return_loss=False,
    test_mode=False,
    worker_mode=False,
    worker_name=None,
):
    average_loss = 0
    correct_predictions = 0
    total_predictions = 0
    targets = []
    predictions = []

    with torch.no_grad():
        model.eval()
        for images, labels in loader:
            outputs = model(images)
            average_loss += loss_func(outputs, labels).item() * images.size(0)
            _, pred = torch.max(outputs, 1)
            correct_predictions += torch.sum(pred == labels.data)

            total_predictions += images.size(0)

            targets.extend(labels.view(-1).tolist())
            predictions.extend(pred.view(-1).tolist())

    average_loss /= total_predictions
    average_accuracy = correct_predictions / total_predictions

    if worker_mode:
        print(f"Train CR of {worker_name}:")
        report = CR(targets, predictions, zero_division=0)
        print(report)

    if test_mode:
        return (
            average_accuracy,
            correct_predictions,
            total_predictions,
            average_loss,
            targets,
            predictions,
        )
    elif return_loss:
        return average_accuracy, correct_predictions, total_predictions, average_loss
    else:
        return average_accuracy, correct_predictions, total_predictions


def get_suffix(
    val,
    alt_model,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    delay,
    slow_worker_1,
    delay_intensity,
    delay_type,
    use_alr,
    lrs,
    saves_per_epoch,
):
    suffix = ""
    if use_alr:
        suffix += f"_ADAM"
    else:
        suffix += f"_SGD"
    if lrs is not None:
        suffix += f"_{lrs}"
    if saves_per_epoch is not None:
        suffix += f"_spe{saves_per_epoch}"
    if val:
        suffix += "_val"
    if alt_model:
        suffix += "_alt_model"
    if split_dataset:
        suffix += "_split_dataset"
    elif split_labels:
        suffix += "_labels"
    elif split_labels_unscaled:
        suffix += "_labels_unscaled"
    if delay:
        suffix += "_delay"
    if slow_worker_1:
        suffix += "_slow_worker_1"
    if delay_intensity is not None:
        suffix += f"_{delay_intensity}"
    if delay_type is not None:
        suffix += f"_{delay_type}"

    return suffix


def get_base_name(
    mode,
    dataset_name,
    world_size,
    train_split,
    learning_rate,
    momentum,
    batch_size,
    epochs,
    val,
    use_alr,
    lrs,
    saves_per_epoch,
    alt_model=False,
    split_dataset=False,
    split_labels=False,
    split_labels_unscaled=False,
    delay=False,
    slow_worker_1=False,
    delay_intensity=None,
    delay_type=None,
):
    suffix = get_suffix(
        val,
        alt_model,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        delay,
        slow_worker_1,
        delay_intensity,
        delay_type,
        use_alr,
        lrs,
        saves_per_epoch,
    )

    base_name = f"{dataset_name}_{mode}_{world_size}_{str(float(train_split)*10).replace('.', '')}_{str(learning_rate).replace('.', '')}_{str(momentum).replace('.', '')}_{batch_size}_{epochs}{suffix}"

    return base_name


def _save_model(
    base_name,
    subfolder,
    model,
):
    filename = base_name + "_model.pt"

    if len(subfolder) > 0:
        filepath = os.path.join(subfolder, filename)
    else:
        filepath = filename

    torch.save(model.state_dict(), filepath)
    print(f"Model saved: {filepath}")


def save_weights(
    base_name,
    subfolder,
    weights_matrix,
):
    filename = base_name + "_weights.npy"

    flat_weights = [
        np.hstack([w.flatten() for w in epoch_weights])
        for epoch_weights in weights_matrix
    ]
    weights_matrix_np = np.vstack(flat_weights)

    if len(subfolder) > 0:
        filepath = os.path.join(subfolder, filename)
    else:
        filepath = filename

    np.save(filepath, weights_matrix_np)
    print(f"Weights {weights_matrix_np.shape} saved: {filepath}")


def compute_weights_l2_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def _delay(intensity=DEFAULT_DELAY_INTENSITY, _type=DEFAULT_DELAY_TYPE, worker_1=False):
    delay_mean, delay_std = DELAY_VALUES[intensity]
    if worker_1:
        delay_mean *= DELAY_WORKER_1_FACTOR
        delay_std *= DELAY_WORKER_1_FACTOR
    if _type == "gaussian":
        delay_time = np.random.normal(delay_mean, delay_std)
        delay_time = max(0, delay_time)  # Ensure the delay is not negative
    elif _type == "constant":
        delay_time = delay_mean
    time.sleep(delay_time)

# constant small 10 ms
# gaussian small 10 ms + 1ms / 0.1ms std

#################################### NET ####################################
class CNN_MNIST(nn.Module):  # LeNet 5 for MNIST and Fashion MNIST
    def __init__(self, loss_func=nn.functional.nll_loss):
        print(f"LeNet5 using loss: {loss_func}")
        super(CNN_MNIST, self).__init__()
        self.loss_func = loss_func
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        if self.loss_func == nn.functional.nll_loss:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class CNN_MNIST_alt(nn.Module):  # PyTorch model for MNIST and Fashion MNIST
    def __init__(self, loss_func=nn.functional.nll_loss):
        super(CNN_MNIST_alt, self).__init__()
        self.loss_func = loss_func
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        print(f"CNN using loss: {loss_func}")

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.loss_func == nn.functional.nll_loss:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out


class ResNet(nn.Module):  # ResNet18
    def __init__(
        self,
        num_classes=10,
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        loss_func=nn.functional.nll_loss,
    ):
        print(f"ResNet18 using loss: {loss_func}")
        super(ResNet, self).__init__()
        self.loss_func = loss_func
        self.in_channels = 64
        self.loss_func = loss_func
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        if self.loss_func == nn.functional.nll_loss:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class CNN_CIFAR10(ResNet):
    def __init__(self, loss_func=nn.functional.nll_loss):
        super().__init__(num_classes=10, loss_func=loss_func)


class CNN_CIFAR100(ResNet):
    def __init__(self, loss_func=nn.functional.nll_loss):
        super().__init__(num_classes=100, loss_func=loss_func)


class CNN_CIFAR10_alt(nn.Module):  # Adapted PyTorch model for CIFAR10
    def __init__(self, loss_func=nn.functional.nll_loss):
        print(f"CNN using loss: {loss_func}")
        super(CNN_CIFAR10_alt, self).__init__()
        self.loss_func = loss_func
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.loss_func == nn.functional.nll_loss:
            x = nn.functional.log_softmax(x, dim=1)
        return x


class CNN_CIFAR100_alt(nn.Module):  # Adapted PyTorch model for CIFAR100
    def __init__(self, loss_func=nn.functional.nll_loss):
        print(f"CNN using loss: {loss_func}")
        super(CNN_CIFAR100_alt, self).__init__()
        self.loss_func = loss_func
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.loss_func == nn.functional.nll_loss:
            x = nn.functional.log_softmax(x, dim=1)
        return x


#################################### Dataloader utility functions ####################################
def count_distinct_labels(dataset):
    labels = dataset.targets
    unique_labels = torch.unique(torch.as_tensor(labels))
    return len(unique_labels)


def get_mean_std(dataset):
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(trainloader))[0]
    mean = data.mean()
    std = data.std()
    return mean, std


def get_shuffled_indices(dataset_length, train_split):
    train_data_indices = torch.randperm(dataset_length)
    train_length = int(train_split * dataset_length)
    subsample_train_indices = train_data_indices[:train_length]
    return subsample_train_indices


def get_batch_size(batch_size, train_length):
    if batch_size < 1 or batch_size > train_length:
        print("Forbidden value !!! batch_size must be between [1,len(train set)]")
        exit()

    return batch_size


#################################### Train loaders types ####################################
def create_default_trainloaders(
    dataset, subsample_train_indices, batch_size, model_accuracy
):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(subsample_train_indices),
    )
    if model_accuracy:
        train_loader_full = train_loader
        return (train_loader, train_loader_full)
    else:
        return train_loader


def create_validation_trainloaders(
    dataset,
    subsample_train_indices,
    batch_size,
    model_accuracy,
    val_split=DEFAULT_VAL_SPLIT,
):
    train_len = int(len(subsample_train_indices) * (1 - val_split))
    val_len = len(subsample_train_indices) - train_len

    train_indices, val_indices = torch.utils.data.random_split(
        subsample_train_indices, [train_len, val_len]
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_indices),
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_indices),
    )
    if model_accuracy:
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(subsample_train_indices),
        )
        return ((train_loader, val_loader), train_loader_full)
    else:
        return (train_loader, val_loader)


def create_split_dataset_trainloaders(
    nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
):
    worker_indices = subsample_train_indices.chunk(nb_workers)
    split_dataset_dataloaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(worker_idx),
        )
        for worker_idx in worker_indices
    ]
    if model_accuracy:
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(subsample_train_indices),
        )
        return (split_dataset_dataloaders, train_loader_full)
    else:
        return split_dataset_dataloaders


def create_split_labels_trainloaders(
    nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
):
    nb_labels = count_distinct_labels(dataset)

    label_indices = {i: [] for i in range(nb_labels)}

    for idx in subsample_train_indices:
        label = dataset[idx][1]
        label_indices[label].append(idx)

    worker_indices = [[] for _ in range(nb_workers)]
    available_labels = list(range(nb_labels))
    np.random.shuffle(available_labels)

    for i, label in enumerate(available_labels):
        worker = i % nb_workers
        worker_indices[worker].extend(label_indices[label])

    min_subset_length = len(subsample_train_indices)
    for worker in worker_indices:
        if min_subset_length > len(worker):
            min_subset_length = len(worker)

    for i, worker in enumerate(worker_indices):
        worker_indices[i] = worker[:min_subset_length]

    labels_train_loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(worker_idx),
        )
        for worker_idx in worker_indices
    ]

    if model_accuracy:
        full_labels_list = []
        for sublist in worker_indices:
            full_labels_list.extend(sublist)
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(full_labels_list),
        )
        return (labels_train_loaders, train_loader_full)
    else:
        return labels_train_loaders


def create_split_labels_unscaled_trainloaders(
    nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
):
    nb_labels = count_distinct_labels(dataset)

    label_indices = {i: [] for i in range(nb_labels)}

    for idx in subsample_train_indices:
        label = dataset[idx][1]
        label_indices[label].append(idx)

    worker_indices = [[] for _ in range(nb_workers)]
    available_labels = list(range(nb_labels))
    np.random.shuffle(available_labels)

    for i, label in enumerate(available_labels):
        worker = i % nb_workers
        worker_indices[worker].extend(label_indices[label])

    labels_train_loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(worker_idx),
        )
        for worker_idx in worker_indices
    ]

    if model_accuracy:
        full_labels_list = []
        for sublist in worker_indices:
            full_labels_list.extend(sublist)
        train_loader_full = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(full_labels_list),
        )
        return (labels_train_loaders, train_loader_full)
    else:
        return labels_train_loaders


def get_trainloader(
    nb_workers,
    dataset,
    subsample_train_indices,
    batch_size,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    model_accuracy,
    validation,
):
    if (
        not split_dataset
        and not split_labels
        and not split_labels_unscaled
        and not validation
    ):
        train_loaders = create_default_trainloaders(
            dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif validation:
        train_loaders = create_validation_trainloaders(
            dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif split_dataset and not split_labels and not split_labels_unscaled:
        train_loaders = create_split_dataset_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders

    elif split_labels and not split_labels_unscaled:
        train_loaders = create_split_labels_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders
    else:
        train_loaders = create_split_labels_unscaled_trainloaders(
            nb_workers, dataset, subsample_train_indices, batch_size, model_accuracy
        )
        return train_loaders


#################################### Dataset train loaders ####################################
def mnist_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
    validation,
):
    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(mnist_train)

    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    subsample_train_indices = get_shuffled_indices(len(mnist_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created MNIST trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        mnist_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        model_accuracy,
        validation,
    )
    return (train_loaders, batch_size)


def fashion_mnist_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
    validation,
):
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(fashion_mnist_train)

    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    subsample_train_indices = get_shuffled_indices(
        len(fashion_mnist_train), train_split
    )
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created FashionMNIST trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        fashion_mnist_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        model_accuracy,
        validation,
    )
    return (train_loaders, batch_size)


def cifar10_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
    validation,
):
    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar10_train)

    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    subsample_train_indices = get_shuffled_indices(len(cifar10_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created CIFAR10 trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        cifar10_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        model_accuracy,
        validation,
    )
    return (train_loaders, batch_size)


def cifar100_trainloaders(
    nb_workers,
    split_dataset,
    split_labels,
    split_labels_unscaled,
    train_split,
    batch_size,
    model_accuracy,
    validation,
):
    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar100_train)

    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
            ]
        ),
    )

    subsample_train_indices = get_shuffled_indices(len(cifar100_train), train_split)
    batch_size = get_batch_size(batch_size, len(subsample_train_indices))
    print("Created CIFAR100 trainloaders")
    train_loaders = get_trainloader(
        nb_workers,
        cifar100_train,
        subsample_train_indices,
        batch_size,
        split_dataset,
        split_labels,
        split_labels_unscaled,
        model_accuracy,
        validation,
    )
    return (train_loaders, batch_size)


#################################### TRAIN LOADER main function ####################################
def create_worker_trainloaders(
    dataset_name,
    train_split,
    batch_size,
    model_accuracy,
    nb_workers=1,
    split_dataset=False,
    split_labels=False,
    split_labels_unscaled=False,
    validation=False,
):
    if dataset_name == "mnist":
        return mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
            validation,
        )
    elif dataset_name == "fashion_mnist":
        return fashion_mnist_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
            validation,
        )
    elif dataset_name == "cifar10":
        return cifar10_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
            validation,
        )
    elif dataset_name == "cifar100":
        return cifar100_trainloaders(
            nb_workers,
            split_dataset,
            split_labels,
            split_labels_unscaled,
            train_split,
            batch_size,
            model_accuracy,
            validation,
        )
    else:
        print("Error Unkown dataset")
        exit()


#################################### Dataset test loaders ####################################
def mnist_testloader(batch_size, test=True):
    mnist_train = torchvision.datasets.MNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(mnist_train)

    if test:
        mnist_test = torchvision.datasets.MNIST(
            "data/",
            download=True,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            mnist_test, batch_size=batch_size, shuffle=False
        )
    else:
        mnist_train = torchvision.datasets.MNIST(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def fashion_mnist_testloader(batch_size, test=True):
    fashion_mnist_train = torchvision.datasets.FashionMNIST(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(fashion_mnist_train)

    if test:
        fashion_mnist_test = torchvision.datasets.FashionMNIST(
            "data/",
            download=True,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            fashion_mnist_test, batch_size=batch_size, shuffle=False
        )
    else:
        fashion_mnist_train = torchvision.datasets.FashionMNIST(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            fashion_mnist_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def cifar10_testloader(batch_size, test=True):
    cifar10_train = torchvision.datasets.CIFAR10(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar10_train)
    if test:
        cifar10_test = torchvision.datasets.CIFAR10(
            "data/",
            download=True,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            cifar10_test, batch_size=batch_size, shuffle=False
        )
    else:
        cifar10_train = torchvision.datasets.CIFAR10(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            cifar10_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


def cifar100_testloader(batch_size, test=True):
    cifar100_train = torchvision.datasets.CIFAR100(
        "data/",
        download=True,
        train=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    mean, std = get_mean_std(cifar100_train)
    if test:
        cifar100_test = torchvision.datasets.CIFAR100(
            "data/",
            download=True,
            train=False,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            cifar100_test, batch_size=batch_size, shuffle=False
        )
    else:
        cifar100_train = torchvision.datasets.CIFAR100(
            "data/",
            download=True,
            train=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((mean.item(),), (std.item(),)),
                ]
            ),
        )
        test_loader = torch.utils.data.DataLoader(
            cifar100_train, batch_size=batch_size, shuffle=False
        )
    return test_loader


#################################### TEST LOADER main function ####################################
def create_testloader(model_path, batch_size):
    if "fashion_mnist" in model_path:
        # print("Created FashionMNIST testloader \n")
        return fashion_mnist_testloader(batch_size)
    elif "mnist" in model_path:
        # print("Created MNIST testloader \n")
        return mnist_testloader(batch_size)
    elif "cifar100" in model_path:
        # print("Created CIFAR100 testloader \n")
        return cifar100_testloader(batch_size)
    elif "cifar10" in model_path:
        # print("Created CIFAR10 testloader \n")
        return cifar10_testloader(batch_size)
    else:
        print("Error Unkown dataset")
        exit()


def create_trainloader(model_path, batch_size):
    if "fashion_mnist" in model_path:
        # print("Created FashionMNIST trainloader \n")
        return fashion_mnist_testloader(batch_size, test=False)
    elif "mnist" in model_path:
        # print("Created MNIST trainloader \n")
        return mnist_testloader(batch_size, test=False)
    elif "cifar100" in model_path:
        # print("Created CIFAR100 trainloader \n")
        return cifar100_testloader(batch_size, test=False)
    elif "cifar10" in model_path:
        # print("Created CIFAR10 trainloader \n")
        return cifar10_testloader(batch_size, test=False)
    else:
        print("Error Unkown dataset")
        exit()


#################################### LOGGER ####################################
def setup_simple_logger(subfolder, base_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_filename = f"{base_name}_log.log"

    # create file handler which logs even debug messages
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        fh = logging.FileHandler(os.path.join(subfolder, log_filename), mode="w")
    else:
        fh = logging.FileHandler(log_filename, mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def setup_logger(log_queue):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    qh = QueueHandler(log_queue)
    qh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    qh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(qh)

    return logger


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


def log_writer(log_queue, subfolder, filename):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if len(subfolder) > 0:
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        with open(os.path.join(subfolder, filename), "w") as log_file:
            while True:
                try:
                    record = log_queue.get(timeout=1)
                    if record is None:
                        break
                    msg = formatter.format(record)
                    log_file.write(msg + "\n")
                except queue.Empty:
                    continue
    else:
        with open(filename, "w") as log_file:
            while True:
                try:
                    record = log_queue.get(timeout=1)
                    if record is None:
                        break
                    msg = formatter.format(record)
                    log_file.write(msg + "\n")
                except queue.Empty:
                    continue
