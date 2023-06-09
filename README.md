# OptML CS-439 EPFL
Optimization for Machine Learning project by Robin Junod, Arthur Lamour and Nicolas Reategui

## Objective

* How do varying delays impact the convergence of distributed asynchronous SGD?
* Does the partitioning of data among workers enhance convergence during the training process? Is it beneficial to distribute labels among workers as well?
* What is the influence of momentum on asynchronous SGD? Does it serve as a regularization factor?

## Background
    
![alt text](https://miro.medium.com/v2/resize:fit:2000/format:webp/1*RWmAPFhueGd4Ec2C_w61JQ.png "Asynchronous SGD vs Synchronous SGD")
Source: [Truly Sparse Neural Networks at Scale](https://www.researchgate.net/publication/348508649_Truly_Sparse_Neural_Networks_at_Scale/download)

Asynchronous SGD: workers update and fetch the global model without waiting for other workers.
Synchronous SGD: workers send their gradient to the parameter server, once the gradients of each worker are received, the parameter server averages them and update the global model and workers fetch the same global model.
Parameter server: master responsible for the global model and  coordination among workers
Workers: training nodes
## Installation
Option 1 (Docker conda OS independant):
In order to avoid compatibility issues, you can use docker. This will allow you to use a lightweight Linux image in your computer for CLI programs which is enough for the scope of this project as we don't need a GUI. First install Docker or [Docker desktop](https://docs.docker.com/desktop/install/windows-install/) which will enable the docker service. Once docker is installed run the following command PS: `docker build --pull --rm -f "docker/Dockerfile" -t optml "docker"  --shm-size=1g` from the repo folder, this will create the image. It is necessary to use a Linux distribution in order to use the **PyTorch RPC** functionalities. After building the image run `docker run -v path_to_your_repo:mount_path --rm --shm-size=5g -it optml` this will initialize the container, mount your repo to the specified path to use it within the container and the `--rm` will remove everything from your container after killing it to avoid waisting memory. To use VS Code with your container, download the Docker extension and once the container is running attach VS to it. Activate the base environment inside the container, to do so you can execute `conda activate`. It may be necessary to execute `conda init bash` in a terminal inside the container and then open a new one to be able to use conda. Once the base environment is activated you can run python.

Option 2 (WSL Windows only):
Use Windows Subsystem for Linux (WSL). To install WSL2 take the following steps:
- Open PowerShell or Windows Command Prompt in administrator and run `wsl install`
- Install the default Ubuntu distribution `wsl --install -d Ubuntu`
- In bash run (**disable firewall** if needed):
  - `sudo apt update`
  - `sudo apt upgrade`
  - `sudo apt install python3-pip dos2unix bc`
  - `pip3 install torch torchvision torchaudio tqdm --index-url https://download.pytorch.org/whl/cpu`
  - `pip3 install matplotlib scikit-learn papermill plotly`

## Files
### Python Scripts
- `nn_train.py`: implements non distributed training
- `dnn_sync_train.py`: simulates distributed synchronous SGD on CPU
- `dnn_async_train.py`: simulates distributed asynchronous SGD on CPU
- `test_model.py`: computes the train and test performance and generates training plots
- `loss_landscape.py`: generates a 3D plot of the loss landscape and 2D contour plot with the training trajectory 
- `loss_landscape_multi_traj.py`: generates a 3D plot of the loss landscape with multiple training trajectories and 2D contour plot with multiple training trajectories
- `kfold.py`: implements grid search on hyperparameters to find the optimal ones 
- `models_size.py`: prints the number of parameters of the available models
- `common.py`: all the common functions used by the other scripts
### Bash Scripts
- loss_landcape.sh
- run_async_momentums.sh
- run_delay_comp.sh
- run_kfold.sh
- run_kfold_full.sh
- run_speed_comp.sh
- run_val.sh
- run_val_full.sh
- test_model.sh

### Results
Summaries of some experiences can be found in the summaries folder, to get access to all the results go to our [GDrive](https://drive.google.com/drive/folders/1rM8yHsevoPhG_gKhCVNJ2S3qwUvvFWgU?usp=sharing) with ~ 10 GB of data including models, weights during training, classification reports, loss landscape plots and contour plots.

## Flags
`nn_train.py`, `dnn_sync_train.py`, `dnn_async_train.py`

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

- The following command will train 5 workers synchronously, on the full MNIST train dataset, trainloaders will use a batch size of 1, and model accuracy will be printed at the end. With the `--split_labels` flag, the 10 digits will be splitted evenly between the workers. For this example, we have 5 workers, meaning that each worker will train on two randomly chosen digits in $\[0,9\]$, here is an illustrative example: `python3 dnn_sync_train.py --dataset mnist --world_size 6 --model_accuracy --batch_size 1 --split_labels`

<div align="center">

| Worker | Digits |
| ---- | ---- |
| 1 | 3, 5 |
| 2 | 9, 2 |
| 3 | 7, 4 |
| 4 | 1, 6 |
| 5 | 0, 8 |

</div>

 

