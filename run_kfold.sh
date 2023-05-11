#!/bin/bash

{
    python3 kfold.py --dataset mnist
    python3 kfold.py --dataset fashion_mnist
    python3 kfold.py --dataset cifar10
    python3 kfold.py --dataset cifar100
} &> kfold.txt