# OptML
Optimization for Machine Learning project by Robin Junod, Arthur Lamour and Nicolas Reategui

# Objective
Asynchronous SGD: What role does momentum plays? Does it act as regularizer as well SGD? 
We can then analyze the shape of the minima using perhaps the two techniques in summary? What impact do different delays or learning schedulers have on the minima and converge time?

Asynchronous SGD: How do different delays affect convergence? How does it interplay with momentum?
Does it act as a regularizer, like drop-out?

## Installation

In order to avoid compatibility issues, you can use docker. This will allow you to use a lightweight Linux image in your computer for CLI programs which is enough for the scope of this project as we don't need a GUI. First install Docker or [Docker desktop](https://docs.docker.com/desktop/install/windows-install/) which will enable the docker service. Once docker is installed run the following command PS `docker build --pull --rm -f "docker/Dockerfile" -t optml "docker"  --shm-size=1g` from the repo folder, this will create the image. It is necessary to use a Linux distribution in order to use the **PyTorch RPC** functionalities. After building the image run `docker run -v path_to_your_repo:mount_path --rm --shm-size=5g -it optml` this will initialize the container, mount your repo to the specified path to use it within the container and the `--rm` will remove everything from your container after killing it to avoid waisting memory. To use VS Code with your container, download the Docker extension and once the container is running attach VS to it.

Windows users can also use Windows Subsystem for Linux (WSL). To install WSL take the following steps:
- Open PowerShell or Windows Command Prompt in administrator and run `wsl install`
- Install the default Ubuntu distribution `wsl --install -d Ubuntu`
- In bash run (**disable firewall** if needed):
  - `sudo apt update`
  - `sudo apt upgrade`
  - `sudo apt install python3-pip`
  - `pip3 install torch torchvision torchaudio tqdm --index-url https://download.pytorch.org/whl/cpu`

## How to use our scripts

To use *dnn_mnist.py* you have to activate the base environment inside the container, to do so you can execute `conda activate`. It may be necessary to execute `conda init bash` in a terminal inside the container and then open a new one to be able to use conda. Once the base environment is activated you can run python *dnn_mnist.py* and you will start training. The python script accepts as arguments *world_size* that you can use to specify the number of nodes for training: $1$ parameter server + $n-1$ workers.

For WSL, run in bash `python3 dnn_mnist.py` or `python3 dnn_mnist.py --world_size N` with $N \geqslant 2$.
