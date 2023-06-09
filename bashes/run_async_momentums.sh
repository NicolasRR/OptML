#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
dnn_async_train_py="$project_dir/dnn_async_train.py"
test_model_py="$project_dir/test_model.py"
loss_landscape_multi_traj_py="$project_dir/loss_landscape_multi_traj.py"
subfolder="$project_dir/Results_AsyncMomentum"

dataset="fashion_mnist"
world_size=4
lr=0.005
batch_size=32
epochs=6
train_split=1
grid_size=100

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs) epochs="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --grid_size) grid_size="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')

momentum=0.99
formatted_momentum=$(echo $momentum | tr -d '.')
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

momentum=0.95
formatted_momentum=$(echo $momentum | tr -d '.')
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


momentum=0.9
formatted_momentum=$(echo $momentum | tr -d '.')
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
echo $model_async
echo $log_async
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


momentum=0.5
formatted_momentum=$(echo $momentum | tr -d '.')
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

momentum=0.0
formatted_momentum=$(echo $momentum | tr -d '.')
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}_SGD_spe3_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $loss_landscape_multi_traj_py --grid_size $grid_size --subfolder $subfolder