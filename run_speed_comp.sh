#!/bin/bash

epochs=6
lr=0.01 # maybe need another lr for alt model: 1e-3
momentum=0.9
batch_size=100
world_size_1=4
world_size_2=7

# non sync 
python3 nn_train.py --dataset fashion_mnist --model_accuracy --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs
sleep 0.1
echo
python3 nn_train.py --dataset fashion_mnist --model_accuracy --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --alt_model
sleep 0.1
echo
# sync
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --alt_model
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --alt_model
sleep 0.1
echo
# sync up world_size
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --alt_model
sleep 0.1
echo
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --alt_model
sleep 0.1
echo
# async
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --alt_model
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --alt_model
sleep 0.1
echo
# async up world_size
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --alt_model
sleep 0.1
echo
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --alt_model
