#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
dnn_sync_train_py="$project_dir/dnn_sync_train.py"
dnn_async_train_py="$project_dir/dnn_async_train.py"
test_model_py="$project_dir/test_model.py"
subfolder="$project_dir/Results_AsyncMomentum"

dataset="fashion_mnist"
world_size=4
lr=0.005 
train_split=1
momentum1=0.0
momentum2=0.5
momentum3=0.9
momentum4=0.95
momentum5=0.99
batch_size=32
epochs=6

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs) epochs="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --train_split) train_split="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum1=$(echo $momentum1 | tr -d '.')
formatted_momentum2=$(echo $momentum2 | tr -d '.')
formatted_momentum3=$(echo $momentum3 | tr -d '.')
formatted_momentum4=$(echo $momentum4 | tr -d '.')
formatted_momentum5=$(echo $momentum5 | tr -d '.')


# async baseline
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum1 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --val --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo
