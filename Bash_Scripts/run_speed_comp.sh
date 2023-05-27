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
model_name="${subfolder}/${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

# sync
python3 dnn_sync_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 dnn_sync_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

# sync up world_size
python3 dnn_sync_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 $test_model_py $model_sync --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 dnn_sync_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_sync_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 $test_model_py $model_sync --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

# async
python3 dnn_async_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_1 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_1}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

# async up world_size
python3 dnn_async_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --model_accuracy --seed --world_size $world_size_2 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size_2}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_split_dataset.pt" 
python3 $test_model_py $model_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo