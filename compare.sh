#!/bin/bash

#dos2unix compare.sh
#bash compare.sh

python3 nn_train.py --train_split 0.5 --epoch 2 --dataset mnist
sleep 0.1
echo
python3 dnn_sync_train.py --world_size 3 --train_split 0.5 --epoch 2 --dataset mnist
sleep 0.1
echo
python3 dnn_async_train.py --world_size 3 --train_split 0.5 --epoch 2 --dataset mnist
sleep 0.1
echo
papermill Compare.ipynb Compare.ipynb -p model_classic mnist_classic_05_0001_00_32_2.pt -p model_sync mnist_sync_3_05_0001_00_32_2.pt -p model_async mnist_async_3_05_0001_00_32_2.pt

#jupyter nbconvert --to notebook --execute my_notebook.ipynb --ExecutePreprocessor.timeout=-1 --output temp.ipynb --allow-errors --Application.log_level=0 --Application.verbose=False --input_file path/to/input/file.csv