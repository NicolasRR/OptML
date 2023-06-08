#!/bin/bash

cd ..

dataset="fashion_mnist"
world_size=4
lr=0.005
batch_size=32
epochs=6
subfolder_name="Results_AsyncMomentum"

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


# async baseline
momentum=0.99
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async baseline
momentum=0.95
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo


# async baseline
momentum=0.9
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo


# async baseline
momentum=0.5
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo

# async baseline
momentum=0
python3 dnn_async_train.py --dataset $dataset --world_size $world_size --model_accuracy --seed  --val --saves_per_epoch 3 --lr $lr --momentum $momentum --batch_size $batch_size --epochs $epochs --subfolder $subfolder_name
sleep 0.1
echo
model_async="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_val.pt"
python3 test_model.py $model_async --classification_report --training_time --pics --subfolder $subfolder_name
sleep 0.1
echo