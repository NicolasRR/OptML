#!/bin/bash

#dos2unix compare.sh
#bash compare.sh

include_model_classic=false
include_classification_report=false

train_split=0.2
epoch=2
world_size=3
lr=0.01
momentum=0.0
batch_size=32
dataset="mnist"

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_classic) include_model_classic=true; shift ;;
    --classification_report) include_classification_report=true; shift ;;
    --train_split) train_split="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

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

if $include_model_classic; then
    if $include_classification_report; then
        papermill Compare.ipynb Compare_out.ipynb -p model_classic $model_classic -p model_sync $model_sync -p model_async $model_async -p include_classification_report $include_classification_report
    else
        papermill Compare.ipynb Compare_out.ipynb -p model_classic $model_classic -p model_sync $model_sync -p model_async $model_async
    fi
else
    if $include_classification_report; then
        papermill Compare.ipynb Compare_out.ipynb -p model_sync $model_sync -p model_async $model_async -p include_classification_report $include_classification_report
    else
        papermill Compare.ipynb Compare_out.ipynb -p model_sync $model_sync -p model_async $model_async
    fi
fi