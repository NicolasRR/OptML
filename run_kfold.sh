#!/bin/bash

if [ "$1" == "--alr" ]
then
    {
        echo "Using Adam as optimizer."
        python3 kfold.py --dataset mnist --alr
        sleep 0.1
        echo
        python3 kfold.py --dataset fashion_mnist --alr
        sleep 0.1
        echo
        # python3 kfold.py --dataset cifar10 --alr
        # sleep 0.1
        # echo
        # python3 kfold.py --dataset cifar100 --alr
        # sleep 0.1
        # echo
    } | tee kfold_alr.txt
else
    {
        echo "Using SGD as optimizer."
        python3 kfold.py --dataset mnist
        sleep 0.1
        echo
        python3 kfold.py --dataset fashion_mnist
        sleep 0.1
        echo
        # python3 kfold.py --dataset cifar10
        # sleep 0.1
        # echo
        # python3 kfold.py --dataset cifar100
        # sleep 0.1
        # echo
    } | tee kfold.txt # |& tee kfold_alr.txt # for stderr capture
fi
