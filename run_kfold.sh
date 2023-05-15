#!/bin/bash

if [ "$1" == "--alr" ]
then
    {
        python3 -u kfold.py --dataset mnist --alr
        sleep 0.1
        echo
        python3 -u kfold.py --dataset fashion_mnist --alr
        sleep 0.1
        echo
        python3 -u kfold.py --dataset cifar10 --alr
        sleep 0.1
        echo
        # python3 -u kfold.py --dataset cifar100 --alr
        # sleep 0.1
        # echo
    } |& tee kfold_alr.txt
else
    {
        # python3 -u kfold.py --dataset mnist --momentum
        # sleep 0.1
        # echo
        # python3 -u kfold.py --dataset fashion_mnist --momentum
        # sleep 0.1
        # echo
        python3 -u kfold.py --dataset cifar10 --momentum
        sleep 0.1
        echo
        # python3 -u kfold.py --dataset cifar100
        # sleep 0.1
        # echo
    } |& tee kfold.txt
fi
