#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
dnn_sync_train_py="$project_dir/dnn_sync_train.py"
dnn_async_train_py="$project_dir/dnn_async_train.py"
test_model_py="$project_dir/test_model.py"
subfolder="$project_dir/Delay_results"

dataset="fashion_mnist"
train_split=1
world_size1=3
world_size2=6
world_size3=11

lr=0.005 
momentum=0.9
batch_size=32
epochs=6

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs) epochs="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
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

delay1="small"
delay2="medium"
delay3="long"
type1="constant"
type2="gaussian"
mode1="--split_labels"
mode2="--split_dataset"


echo "Delay, splitting and Worldsize experiments"

for world_size in $world_size1 $world_size2 $world_size3; do
  for delay in $delay1 $delay2 $delay3; do
    for mode in $mode1 $mode2; do
      echo $world_size $delay $mode
      python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type1 $mode 
      sleep 0.1
      echo
      echo $world_size $delay $mode "slow worker1"
      python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type1 $mode --slow_worker_1
      sleep 0.1
      echo
done
done
done

echo "Compensation experiments"

# Manually change the lambda in the corresponding file 

for world_size in $world_size1 $world_size2 $world_size3; do
  for delay in $delay1 $delay2 $delay3; do
    for mode in $mode1; do

      echo $world_size $delay $mode "slow worker1" "compensation"
      python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type1 $mode --slow_worker_1 --comp
      sleep 0.1
      echo
done
done

echo "Momentum experiments"

for world_size in $world_size1 $world_size2 $world_size3; do
  for delay in $delay1 $delay2 $delay3; do
    for mode in $mode1; do
      for momentum in 0.0 0.5 0.99; do
        echo $world_size $delay $mode "slow worker1"
        python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type1 $mode --slow_worker_1
        sleep 0.1
        echo
done
done
done
done

echo "Momentum, Worldsize, splitting and compensation for Gaussian delays comparison"

for world_size in $world_size1 $world_size2 $world_size3; do
  for delay in $delay1 $delay2 $delay3; do
    for mode in $mode1 $mode2; do
      for momentum in 0.9 0.0 0.5 0.99; do
        echo $world_size $delay $mode $momentum
        python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type2 $mode
        sleep 0.1
        echo
        echo $world_size $delay $mode $momentum "--comp"
        python3 $dnn_async_train_py --dataset $dataset --world_size $world_size --model_accuracy --seed --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder --train_split $train_split --saves_per_epoch 3 --delay --delay_intensity $delay --delay_type $type2 $mode --comp
        sleep 0.1
        echo
done
done
done
done
