#!/bin/bash

dataset="fashion_mnist"
world_size=4
lr=0.005 
momentum=0.9
batch_size=32
epochs=30
subfolder_name="Delay_results"

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

formatted_train_split=$(echo $train_split | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')

# sync
python3 dnn_sync_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_sync="${subfolder}/${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_sync --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async baseline
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# constant delays
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# gaussian delays
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_small_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_medium_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --delay --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_delay_long_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# worker1 constant delay
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "constant"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_constant.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# worker1 gaussian delay
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "small" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_small_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "medium" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_medium_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name --slow_worker_1 --delay_intensity "long" --delay_type "gaussian"
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val_slow_worker_1_long_gaussian.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo