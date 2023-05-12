#!/bin/bash

if [ "$1" == "--alr" ]
then
    {
        python3 kfold.py --dataset mnist --alr
        python3 kfold.py --dataset fashion_mnist --alr
        # python3 kfold.py --dataset cifar10 --alr
        # python3 kfold.py --dataset cifar100 --alr
    } &> kfold_alr.txt
else
    {
        python3 kfold.py --dataset mnist
        python3 kfold.py --dataset fashion_mnist
        # python3 kfold.py --dataset cifar10
        # python3 kfold.py --dataset cifar100
    } &> kfold.txt
fi