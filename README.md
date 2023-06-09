# OptML CS-439 EPFL
Optimization for Machine Learning project by Robin Junod, Arthur Lamour and Nicolas Reategui

## Objective

* How do varying delays impact the convergence of distributed asynchronous SGD?
* Does the partitioning of data among workers enhance convergence during the training process? Is it beneficial to distribute labels among workers as well?
* What is the influence of momentum on asynchronous SGD? Does it serve as a regularization factor?

## Background

<div align="center">

<img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/1*RWmAPFhueGd4Ec2C_w61JQ.png" width="1000">

</div>

Source: [Truly Sparse Neural Networks at Scale](https://www.researchgate.net/publication/348508649_Truly_Sparse_Neural_Networks_at_Scale/download)

- Asynchronous SGD: workers update and fetch the global model without waiting for other workers. <br>
- Synchronous SGD: workers send their gradient to the parameter server, once the gradients of each worker are received, the parameter server averages them and update the global model and workers fetch the same global model. 
- Parameter server: master responsible for the global model and  coordination among workers. 
- Workers: training nodes.
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

Here below is the data flow diagram:
<div align="center">
  
<img src="https://i.postimg.cc/JnvgvSrW/image-2023-06-09-172517991.png" width="575">

</div>

### Bash Scripts
- `loss_landcape.sh`:
- `run_async_momentums.sh`:
- `run_delay_comp.sh`:
- `run_kfold.sh`:
- `run_kfold_full.sh`:
- `run_speed_comp.sh`:
- `run_val.sh`:
- `run_val_full.sh`:
- `test_model.sh`:

### Results
Summaries of some experiences can be found in the summaries folder, to get access to all the results go to our [GDrive](https://drive.google.com/drive/folders/1rM8yHsevoPhG_gKhCVNJ2S3qwUvvFWgU?usp=sharing) with ~ 10 GB of data including models, weights during training, classification reports, loss landscape plots and contour plots.

## Flags
Available flags for `nn_train.py`, `dnn_sync_train.py`, `dnn_async_train.py`:

<div align="center">

| Flag | Description |Note |
| --------------- | --------------- | --------------- |
|--dataset {mnist,fashion_mnist,cifar10,cifar100}|Choose a dataset to train on: mnist, fashion_mnist, cifar10, or cifar100.||
|--train_split TRAIN_SPLIT |Fraction of the training dataset to be used for training (0,1].  | |
| --lr LR |Learning rate of optimizer (0,+inf).  | |
| --momentum MOMENTUM | Momentum of SGD optimizer [0,+inf). | Not used with Adam optimizer.|
| --batch_size BATCH_SIZE  | Batch size of Mini batch SGD [1,len(train set)]. |  |
| --epochs EPOCHS  | Number of epochs for training [1,+inf). |  |
| --model_accuracy |If set, will compute the train accuracy of the global model after training.  |  |
| --no_save_model  |  If set, the trained model will not be saved. |  |
| --seed  | If set, it will set seeds on torch and numpy for reproducibility purposes. |  |
| --subfolder SUBFOLDER |  Subfolder where the model and log.log will be saved. |  |
|--saves_per_epoch SAVES_PER_EPOCH| Number of times the model weights will be saved during one epoch. | The first weights are saved directly after initialization (model has not trained yet). |
|   --alr |  If set, use adaptive learning rate (Adam optimizer) instead of SGD optimizer. |  |
|   --lrs {exponential,cosine_annealing} |  Applies a learning rate scheduler: exponential or cosine_annealing. |  |
|  --alt_model | Will train using alternate CNN models instead of LeNet5 (MNIST & FashionMNIST) or ResNet18 (CIFAR10 & CIFAR100). |  |
|  --val | If set, will create a validation dataloader and compute the loss and accuracy of train set and val set at the end of each epoch. | For asynchronous, the number of batches received (pseudo epochs) are used instead of epochs.|
|  --master_port MASTER_PORT |  Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port. | Not avaiable for `nn_train.py`. |
|   --master_addr MASTER_ADDR | Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port. | Not avaiable for `nn_train.py`. |
|  --world_size WORLD_SIZE | Total number of participating processes. Should be the sum of master node and all training nodes [2,+inf]. | Not avaiable for `nn_train.py`. If `world_size` exceeds the number of available CPU threads, PyTorch RPC will crash.|
|  --delay | Add a delay to all workers at each mini-batch update. | Not avaiable for `nn_train.py`. |
|  --slow_worker_1 | Add a longer delay only to worker 1 at each mini-batch update. | Not avaiable for `nn_train.py`. |
|   --delay_intensity {small,medium,long} |  Applies a delay intensity of: small 10ms, medium 20ms, long 30ms. | Not avaiable for `nn_train.py`. |
|   --delay_type {constant,gaussian} | Applies a delay of type: constant or gaussian. | Not avaiable for `nn_train.py`. |
|   --split_dataset | After applying train_split, each worker will train on a unique distinct dataset (samples will not be shared between workers). Do not use with --split_labels or --split_labels_unscaled. | Not avaiable for `nn_train.py`. |
|   --split_labels |If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. Workers will not share samples and the labels are randomly assigned. Don't use with --split_dataset or --split_labels_unscaled. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded. | Not avaiable for `nn_train.py`. |
|  --split_labels_unscaled | If set, it will split the dataset in {world_size -1} parts, each part corresponding to a distinct set of labels, and each part will be assigned to a worker. Workers will not share samples and the labels are randomly assigned. Note, the training length will be the DIFFERENT for all workers, based on the number of samples each class has. Don't use --split_dataset or split_labels. Depending on the chosen dataset the --world_size should be total_labels mod (world_size-1) = 0, with world_size = 2 excluded. | Not avaiable for `nn_train.py` and `dnn_sync_train.py`. |
| --compensation | If set, enables delay compensation. | Not avaiable for `nn_train.py` and `dnn_sync_train.py`.  |
</div>

For `kfold.py `, `test_model.py`, `loss_landscape.py`, `loss_landscape_multi_traj.py` use `-h` or `--help` to see the available flags and required arguments to give. For example: `python3 kfold.py -h` or `python3 test_model.py --help`.
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

 

