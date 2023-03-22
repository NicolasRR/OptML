FROM ubuntu:18.04
FROM nvidia/cuda:12.1.0-base-ubuntu18.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y software-properties-common &&\
    apt-get update &&\
    apt-get install -y wget build-essential python3.10 git

ENV CONDA_DIR /opt/conda 
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -y -c conda-forge tqdm && \
    conda install -y matplotlib



