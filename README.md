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
| --master_port MASTER_PORT | Row 1, Column 2 |
| --master_addr MASTER_ADDR | Row 2, Column 2 |
| --world_size WORLD_SIZE | Row 3, Column 2 |
| --train_split TRAIN_SPLIT| Row 3, Column 2 |
| --lr LR | Row 3, Column 2 |
| --momentum MOMENTUM | Row 3, Column 2 |
| --batch_size BATCH_SIZE| Row 3, Column 2 |     
| --no_save_model | Row 3, Column 2 |
| --unique_datasets | Row 3, Column 2 |
| --epochs EPOCHS | Row 3, Column 2 |
| --model_accuracy | Row 3, Column 2 |
| --worker_accuracy | Row 3, Column 2 |
| --digits DIGITS | Row 3, Column 2 |