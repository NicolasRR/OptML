#!/bin/bash

#dos2unix compare.sh
#bash compare.sh

train_split=0.5
epoch=2
world_size=3
lr=0.001
momentum=0.0
batch_size=32
dataset="mnist"

# Format the model filenames based on input parameters
formatted_train_split=$(echo $train_split | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')
model_classic="mnist_classic_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
model_sync="mnist_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
model_async="mnist_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"

python3 nn_train.py --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed
sleep 0.1
echo
python3 dnn_sync_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed 
sleep 0.1
echo
python3 dnn_async_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed
sleep 0.1
echo

papermill Compare.ipynb Compare.ipynb -p model_classic $model_classic -p model_sync $model_sync -p model_async $model_async