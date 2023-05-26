#!/bin/bash

epochs=6
lr=0.01 # maybe need another lr for alt model: 1e-3
momentum=0.9
batch_size=100
world_size_1=4
world_size_2=7
subfolder_name="Speed_results"

formatted_train_split=$(echo $train_split | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')


# non sync 
python3 nn_train.py --dataset fashion_mnist --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# sync
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# sync up world_size
python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 test_model.py $model_sync --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_sync_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 test_model.py $model_sync --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async up world_size
python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset fashion_mnist --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder_name --saves_per_epoch 3
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 test_model.py $model_name --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo