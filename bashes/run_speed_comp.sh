#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
nn_train_py="$project_dir/nn_train.py"
dnn_sync_train_py="$project_dir/dnn_sync_train.py"
dnn_async_train_py="$project_dir/dnn_async_train.py"
test_model_py="$project_dir/test_model.py"
subfolder="$project_dir/Speed_results"

dataset="fashion_mnist"
epochs=6 # from !summary_kfold_.txt
lr=0.005
momentum=0.9
batch_size=32
world_size_1=4 # 3 workers
world_size_2=7 # 6 workers
train_split=1

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs) epochs="$2"; shift 2 ;;
    --world_size_1) world_size_1="$2"; shift 2 ;;
    --world_size_2) world_size_2="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --train_split) train_split="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')


# non sync 
python3 $nn_train_py --dataset $dataset --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo

mode2=""
mode1="--split_dataset"
mode3="--split_labels"

for file in $dnn_sync_train_py $dnn_async_train_py; do
  for mode in $mode1 $mode2 $mode3;do
    for world_size in $world_size_1 $world_size_2; do
      python3 $file --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split $mode
      sleep 0.1
      echo
done
done
done
