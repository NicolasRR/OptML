# OptML
Optimization for Machine Learning project by Robin Junod, Arthur Lamour and Nicolas Reategui

# Objective
Asynchronous SGD: What role does momentum plays? Does it act as regularizer as well SGD? 
We can then analyze the shape of the minima using perhaps the two techniques in summary? What impact do different delays or learning schedulers have on the minima and converge time?

Asynchronous SGD: How do different delays affect convergence? How does it interplay with momentum?
Does it act as a regularizer, like drop-out?

Local minima for deep learning: Can you find differences between the ‘shape’ of local minima that SGD
finds, depending on different step-sizes or mini-batch size, vs e.g. AdaGrad or full gradient descent?

## Installation
Option 1 (Docker conda OS independant):
In order to avoid compatibility issues, you can use docker. This will allow you to use a lightweight Linux image in your computer for CLI programs which is enough for the scope of this project as we don't need a GUI. First install Docker or [Docker desktop](https://docs.docker.com/desktop/install/windows-install/) which will enable the docker service. Once docker is installed run the following command PS `docker build --pull --rm -f "docker/Dockerfile" -t optml "docker"  --shm-size=1g` from the repo folder, this will create the image. It is necessary to use a Linux distribution in order to use the **PyTorch RPC** functionalities. After building the image run `docker run -v path_to_your_repo:mount_path --rm --shm-size=5g -it optml` this will initialize the container, mount your repo to the specified path to use it within the container and the `--rm` will remove everything from your container after killing it to avoid waisting memory. To use VS Code with your container, download the Docker extension and once the container is running attach VS to it. Activate the base environment inside the container, to do so you can execute `conda activate`. It may be necessary to execute `conda init bash` in a terminal inside the container and then open a new one to be able to use conda. Once the base environment is activated you can run python.

Option 2 (WSL pip Windows only):
Use Windows Subsystem for Linux (WSL). To install WSL take the following steps:
- Open PowerShell or Windows Command Prompt in administrator and run `wsl install`
- Install the default Ubuntu distribution `wsl --install -d Ubuntu`
- In bash run (**disable firewall** if needed):
  - `sudo apt update`
  - `sudo apt upgrade`
  - `sudo apt install python3-pip`
  - `pip3 install torch torchvision torchaudio tqdm --index-url https://download.pytorch.org/whl/cpu`
  - `pip3 install matplotlib scikit-learn papermill`

## How to use our scripts
We created bash scripts to run and test the SGD variants together effectively. `compare.sh` will run and test the variants one time, `compare_loop.sh` will do the same but multiple times and loops other predefined
couples of parameters such as the number of workers, the learning rate, momentum, ... Both scripts accept various command lines arguments, here is the list:
- `bash compare.sh [flags]`

<div align="center">

| Flag | Description |
| --------------- | --------------- |
| --model_classic | To include vanilla non distributed SGD to compare with distributed synchronous parallel SGD and asynchronous parallel SGD. |
| --classification_report | To include a classification report at testing, usefull for class performance analysis. |
| --notebook | Uses a jupyter notebook to compute the training time for each variants, produce plots, and tests the models performance. |
| --dataset {mnist,fashion_mnist,cifar10,cifar100} | Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100. |
| --world_size WORLD_SIZE | Total number of participating processes. Should be the sum of master node and all training nodes [2,+inf]. |
| --train_split TRAIN_SPLIT| Fraction of the training dataset to be used for training (0,1&#93;. |
| --lr LR | Learning rate of SGD  (0,+inf)." |
| --momentum MOMENTUM | Momentum of SGD  &#91;0,+inf). |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |     
| --epochs EPOCHS | Number of epochs for training &#91;1,+inf&#41;. |

</div>

- `bash compare_loop.sh`

<div align="center">

| Flag | Description |
| --------------- | --------------- |
| --model_classic | To include vanilla non distributed SGD to compare with distributed synchronous parallel SGD and asynchronous parallel SGD. |
| --classification_report | To include a classification report at testing, usefull for class performance analysis. |
| --dataset {mnist,fashion_mnist,cifar10,cifar100} | Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100. |
| --world_size WORLD_SIZE | Total number of participating processes. Should be the sum of master node and all training nodes [2,+inf]. |
| --train_split TRAIN_SPLIT| Fraction of the training dataset to be used for training (0,1&#93;. |
| --lr LR | Learning rate of SGD  (0,+inf)." |
| --momentum MOMENTUM | Momentum of SGD  &#91;0,+inf). |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |     
| --epochs EPOCHS | Number of epochs for training &#91;1,+inf&#41;. |

</div>

- Here is how to run `nn_train.py` (vanilla non distributed SGD): `python3 nn_train.py [flags]`

<div align="center">

| Flag | Description |
| --------------- | --------------- |
| --dataset {mnist,fashion_mnist,cifar10,cifar100} | Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100. |
| --train_split TRAIN_SPLIT| Fraction of the training dataset to be used for training (0,1&#93;. |
| --lr LR | Learning rate of SGD  (0,+inf)." |
| --momentum MOMENTUM | Momentum of SGD  &#91;0,+inf). |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |     
| --epochs EPOCHS | Number of epochs for training &#91;1,+inf&#41;. |
| --model_accuracy | If set, will compute the train accuracy of the global model after training. |
| --no_save_model | If set, the trained model will not be saved. |
| --seed | If set, it will set seeds on `torch`, `numpy` and `random` for reproducibility purposes. |
| --subfolder SUBFOLDER | Subfolder where the model and log.log will be saved. |

</div>

- Here is how to run `dnn_sync_train.py` (synchronous parallel SGD) or `dnn_async_train.py` (asynchronous parallel SGD): `python3 dnn_sync_train.py [flags]` `python3 dnn_async_train.py [flags]`

<div align="center">

| Flag | Description |
| --------------- | --------------- |
| --master_port MASTER_PORT | Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port. |
| --master_addr MASTER_ADDR | Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port. |
| --dataset {mnist,fashion_mnist,cifar10,cifar100} | Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100. |
| --world_size WORLD_SIZE | Total number of participating processes. Should be the sum of master node and all training nodes [2,+inf]. |
| --split_dataset | After applying train_split, each worker will train on a unique distinct dataset (samples will not be shared between workers). |
| --split_labels | If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. Workers will not share samples and the labels are randomly assigned.  The training length will be the **<u>different</u>** for all workers (like in synchronous SGD). This mode requires --batch_size 1, don't use --split_dataset and --split_labels_unscaled. Depending on the chosen dataset the --world_size should be total_labels $mod$ (world_size-1) = 0, with world_size = 2 excluded. |
| --split_labels_unscaled | If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. Workers will not share samples and the labels are randomly assigned. The training length will be **<u>same</u>** for all workers, based on the number of samples each class has. This mode requires --batch_size 1, don't use --split_dataset and --split_labels. Depending on the chosen dataset the --world_size should be total_labels $mod$ (world_size-1) = 0, with world_size = 2 excluded. ***Only available for asynchronous.***|
| --train_split TRAIN_SPLIT| Fraction of the training dataset to be used for training (0,1&#93;. |
| --lr LR | Learning rate of SGD  (0,+inf)." |
| --momentum MOMENTUM | Momentum of SGD  &#91;0,+inf). |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |     
| --epochs EPOCHS | Number of epochs for training &#91;1,+inf&#41;. |
| --model_accuracy | If set, will compute the train accuracy of the global model after training. |
| --worker_accuracy | If set, will compute the train accuracy of each worker after training (useful when --split_dataset). |
| --no_save_model | If set, the trained model will not be saved. |
| --seed | If set, it will set seeds on `torch`, `numpy` and `random` for reproducibility purposes. |
| --subfolder SUBFOLDER | Subfolder where the model and log.log will be saved. |

</div>

- Here is how to run `test_model.py` (compute training time, plots and test performance): `python test_model.py [flags]`

<div align="center">

| Flag | Description |
| --------------- | --------------- |
| --classification_report | To include a classification report at testing, usefull for class performance analysis. |
| --training_time | If set, will read the associated log file to compute the training time. |
| --pics | If set, will compute and save plots from the .log file. |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |   
| --subfolder SUBFOLDER | Subfolder where the model and log.log will be saved. |

</div>


To see all the flags available for a script: `python3 nn_train.py -h` or `python3 test_model.py --help`


### Some examples
- The following command will train two workers synchronously, on 10% of MNIST train dataset, trainloaders will use a batch size of 64, and the SGD optimizer a learning rate of $10^{-2}$ and momentum of $0.5$, at the end of training the global accuracy of the model will be printed and the model will not be saved. As `--epochs EPOCHS` is not precised, the default value of epochs will be used: $1$. <br>
`python3 dnn_sync_train.py --dataset mnist --world_size 3 --train_split 0.1 --lr 0.01 --momentum 0.5 --batch_size 64 --model_accuracy --no_save_model --dataset mnist`

- The following command will train 3 workers synchronously, on 50% of MNIST train dataset, trainloaders will split the 50% of trainset in 3 equal parts for each worker ($60k \Rightarrow 30k \Rightarrow 10k$, $10k$, $10k$), in this configuration workers will not **share** samples (each worker has its distinct trainset). At the end of training, the accuracy of each worker, and global model will be printed, and the model will be saved. <br>
`python3 dnn_sync_train.py --dataset mnist --train_split 0.5 --model_accuracy --worker_accuracy --dataset mnist` 

- The following command will train 5 workers synchronously, on the full MNIST train dataset, trainloaders will use a batch size of 1, and model accuracy will be printed at the end. With the `--split_labels` flag, the 10 digits will be splitted evenly between the workers. For this example, we have 5 workers, meaning that each worker will train on two randomly chosen digits in $\[0,9\]$, here is an illustrative example:

<div align="center">

| Worker | Digits |
| ---- | ---- |
| 1 | 3, 5 |
| 2 | 9, 2 |
| 3 | 7, 4 |
| 4 | 1, 6 |
| 5 | 0, 8 |

</div>

  `python3 dnn_sync_train.py --dataset mnist --world_size 6 --model_accuracy --batch_size 1 --split_labels`

