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
lr_sgd_l5=0.005
lr_sgd_pc=0.01 # pc sgd should be 0.01
lr_alr_l5=0.001 # l5 alr should be 0.001
lr_alr_pc=0.0005
momentum1=0.0
momentum2=0.9
batch_size=32
world_size=6 # 3 workers
train_split=1

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --epochs) epochs="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --train_split) train_split="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr_sgd_l5=$(echo $lr_sgd_l5 | tr -d '.')
formatted_lr_sgd_pc=$(echo $lr_sgd_pc | tr -d '.')
formatted_lr_alr_l5=$(echo $lr_alr_l5 | tr -d '.')
formatted_lr_alr_pc=$(echo $lr_alr_pc | tr -d '.')
formatted_momentum1=$(echo $momentum1 | tr -d '.')
formatted_momentum2=$(echo $momentum2 | tr -d '.')


# async Lenet5 SGD momentum=0.0 1/6
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async Lenet5 SGD momentum=0.9 2/6
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum2 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_val_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_val_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_l5 --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_l5}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async Lenet5 ADAM momentum=0.0 (dosen't matter) 3/6
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split --alr
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_val_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_val_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_l5 --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_l5}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async PC SGD momentum=0.0 4/6
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_alt_model_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_val_alt_model_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_alt_model_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_alt_model_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_alt_model_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum1}_${batch_size}_${epochs}_SGD_spe3_alt_model_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async PC SGD momentum=0.9 5/8
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split --alt_model
sleep 0.1 
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_val_alt_model_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_val_alt_model_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_alt_model_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_alt_model_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_sgd_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_alt_model_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_sgd_pc}_${formatted_momentum2}_${batch_size}_${epochs}_SGD_spe3_alt_model_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async PC ADAM momentum=0.0 (doesn't matter) 6/6
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_val_alt_model_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_val_alt_model_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_alt_model_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_alt_model_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum1 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_alt_model_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum1}_${batch_size}_${epochs}_ADAM_spe3_alt_model_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo


# async PC ADAM momentum=0.9 8/8
python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --subfolder $subfolder --saves_per_epoch 3 --val --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_val_alt_model_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_val_alt_model_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_dataset --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_alt_model_split_dataset_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_alt_model_split_dataset_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo

python3 $dnn_async_train_py --dataset $dataset --model_accuracy --seed --world_size $world_size --lr $lr_alr_pc --momentum $momentum2 --batch_size $batch_size --epochs $epochs --split_labels --subfolder $subfolder --saves_per_epoch 3 --train_split $train_split --alr --alt_model
sleep 0.1
echo
model_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_alt_model_labels_model.pt" 
log_name="${subfolder}/${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr_alr_pc}_${formatted_momentum2}_${batch_size}_${epochs}_ADAM_spe3_alt_model_labels_log.log" 
python3 $test_model_py $model_name $log_name --classification_report --training_time --pics --subfolder $subfolder
sleep 0.1
echo













