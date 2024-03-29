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
world_size_1=3 # 2 workers
world_size_2=6 # 5 workers
world_size_3=11 # 10 workers
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

mode1="--split_dataset"
mode2="--split_labels"

for file in $dnn_sync_train_py $dnn_async_train_py; do
  for mode in $mode1 $mode2;do
    for world_size in $world_size_1 $world_size_2 $world_size_3; do
        echo $file $mode $world_size $model2
        python3 $file --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split $mode
        sleep 0.1
        echo 
done
done
done

# Alt_model

for file in $dnn_sync_train_py $dnn_async_train_py; do
  for mode in $mode1 $mode2;do
    for world_size in $world_size_1 $world_size_2 $world_size_3; do
        echo $file $mode $world_size $model2 "--alt_model"
        python3 $file --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split $mode --alt_model
        sleep 0.1
        echo 
done
done
done

# Classic model and no_splitting

for file in $dnn_sync_train_py $dnn_async_train_py; do
    for world_size in $world_size_1 $world_size_2 $world_size_3; do
        echo $file  $world_size 
        python3 $file --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
        sleep 0.1
        echo 
done
done

# Alt model and no_splitting

for file in $dnn_sync_train_py $dnn_async_train_py; do
    for world_size in $world_size_1 $world_size_2 $world_size_3; do
        echo $file  $world_size "--alt_model"
        python3 $file --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alt_model
        sleep 0.1
        echo 
done
done
