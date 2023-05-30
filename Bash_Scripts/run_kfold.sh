#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
kfold_py="$project_dir/kfold.py"

if [ "$1" == "--alr" ]
then
    {
        # python3 -u $kfold_py --dataset mnist --alr
        # sleep 0.1
        # echo
        python3 -u $kfold_py --dataset fashion_mnist --alr --alt_model
        sleep 0.1
        echo
        # python3 -u $kfold_py --dataset cifar10 --alr
        # sleep 0.1
        #echo
        # python3 -u kfold_py --dataset cifar100 --alr
        # sleep 0.1
        # echo
    } |& tee "$project_dir/kfold_alr.txt"
else
    {
        # python3 -u kfold_py --dataset mnist --momentum
        # sleep 0.1
        # echo
        python3 -u $kfold_py --dataset fashion_mnist --alt_model
        sleep 0.1
        echo
        # python3 -u $kfold_py --dataset cifar10 --momentum
        # sleep 0.1
        # echo
        # python3 -u $kfold_py --dataset cifar100
        # sleep 0.1
        # echo
    } |& tee "$project_dir/kfold.txt"
fi
