#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
dnn_sync_train_py="$project_dir/dnn_sync_train.py"
dnn_async_train_py="$project_dir/dnn_async_train.py"
test_model_py="$project_dir/test_model.py"
subfolder_name="$project_dir/Delay_results"

dataset="fashion_mnist"
train_split=1
world_size=4
lr=0.005 
momentum=0.9
batch_size=32
epochs=30

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epoch) epoch="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')

# sync
python3 $dnn_sync_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_sync="${subfolder}/${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_model.pt"
log_sync="${subfolder}/${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_log.log"
python3 $test_model_py $model_sync $log_sync --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async baseline
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# constant delays
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# gaussian delays
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# worker1 constant delay
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_constant_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_constant_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# worker1 gaussian delay
python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_gaussian_model.pt"
log_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_gaussian_log.log"
python3 $test_model_py $model_async $log_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo