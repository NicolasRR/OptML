#!/bin/bash

dataset="fashion_mnist"
world_size=4
lr=0.005 
momentum=0.9
batch_size=32
epochs=50
subfolder_name="Delay_results"

# sync
python3 dnn_sync_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo

# async baseline
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo

# constant delays
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo

# gaussian delays
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo

# worker1 constant delay
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo

# worker1 gaussian delay
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo