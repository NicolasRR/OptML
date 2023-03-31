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
In order to avoid compatibility issues, you can use docker. This will allow you to use a lightweight Linux image in your computer for CLI programs which is enough for the scope of this project as we don't need a GUI. First install Docker or [Docker desktop](https://docs.docker.com/desktop/install/windows-install/) which will enable the docker service. Once docker is installed run the following command PS `docker build --pull --rm -f "docker/Dockerfile" -t optml "docker"  --shm-size=1g` from the repo folder, this will create the image. It is necessary to use a Linux distribution in order to use the **PyTorch RPC** functionalities. After building the image run `docker run -v path_to_your_repo:mount_path --rm --shm-size=5g -it optml` this will initialize the container, mount your repo to the specified path to use it within the container and the `--rm` will remove everything from your container after killing it to avoid waisting memory. To use VS Code with your container, download the Docker extension and once the container is running attach VS to it.

Option 2 (WSL pip Windows only):
Use Windows Subsystem for Linux (WSL). To install WSL take the following steps:
- Open PowerShell or Windows Command Prompt in administrator and run `wsl install`
- Install the default Ubuntu distribution `wsl --install -d Ubuntu`
- In bash run (**disable firewall** if needed):
  - `sudo apt update`
  - `sudo apt upgrade`
  - `sudo apt install python3-pip`
  - `pip3 install torch torchvision torchaudio tqdm --index-url https://download.pytorch.org/whl/cpu`
  - `pip3 install matplotlib scikit-learn`

## How to use our scripts
Option 1:
To use *dnn_mnist.py* you have to activate the base environment inside the container, to do so you can execute `conda activate`. It may be necessary to execute `conda init bash` in a terminal inside the container and then open a new one to be able to use conda. Once the base environment is activated you can run python *dnn_mnist.py* and you will start training. The python script accepts as arguments *world_size* that you can use to specify the number of nodes for training: $1$ parameter server + $n-1$ workers.

Option 2:
For WSL, run in bash `python3 dnn_mnist.py` or `python3 dnn_mnist.py --world_size N` with $N \geqslant 2$.

## Command line flags
Our scripts accept various arguments from the command line, to see all the flags available: `python3 dnn_mnist.py -h` or `python3 dnn_mnist.py --help`

| Flag | Description |
| --------------- | --------------- |
| --master_port MASTER_PORT | Port that master is listening on, will default to 29500 if not provided. Master must be able to accept network traffic on the host and port. |
| --master_addr MASTER_ADDR | Address of master, will default to localhost if not provided. Master must be able to accept network traffic on the address + port. |
| --world_size WORLD_SIZE | Total number of participating processes. Should be the sum of master node and all training nodes [2,+inf]. |
| --train_split TRAIN_SPLIT| Percentage of the training dataset to be used for training (0,1&#93;. |
| --lr LR | Learning rate of SGD  (0,+inf)." |
| --momentum MOMENTUM | Momentum of SGD  &#91;0,+inf). |
| --batch_size BATCH_SIZE| Batch size of Mini batch SGD [1,len(train set)]. |     
| --no_save_model | If set, the trained model will not be saved. |
| --unique_datasets | After applying train_split, each worker will train on a unique distinct dataset (samples will not be shared between workers). |
| --epochs EPOCHS | Number of epochs for training &#91;1,+inf&#41;. |
| --model_accuracy | If set, will compute the train accuracy of the global model after training. |
| --worker_accuracy | If set, will compute the train accuracy of each worker after training (useful when --unique_datasets). |
| --digits DIGITS | Reprensents the amount of digits that will be trained in parallel, it will split the MNIST dataset in {digits} parts, one part per digit, and each part will be assigned to a worker. This mode requires --world_size {digits +1} --batch_size 1, don't use --unique_datasets. |