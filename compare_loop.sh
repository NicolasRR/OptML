#!/bin/bash

#dos2unix compare_loop.sh
#bash compare_loop.sh > loop_result.txt

include_model_classic=false # include classic sgd to compare with sync and async
include_classification_report=false # include labels classification report

create_subfolder() {
  subfolder_name="experience_"
  i=1
  while true; do
    folder="${subfolder_name}${i}"
    if [ ! -d "$folder" ]; then
      mkdir "$folder"
      break
    fi
    i=$((i+1))
  done
  echo "$folder"
}

#train_split=0.5
epoch=1
#world_size=3
lr=0.01
momentum=0.0
batch_size=32
dataset="mnist"

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_classic) include_model_classic=true; shift ;;
    --classification_report) include_classification_report=true; shift ;;
    #--train_split) train_split="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    #--world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

#world_sizes=(3 4 5 6 7)
world_sizes=(3 4)
#train_splits=(0.2 0.5 1)
train_splits=(0.2 0.5)

test_model_flags=""
if $include_classification_report; then
    test_model_flags+=" --classification_report"
fi
test_model_flags+=" --training_time --pics"

# Nested loops for world_size and train_split
for world_size in "${world_sizes[@]}"; do
  for train_split in "${train_splits[@]}"; do   
    subfolder=$(create_subfolder)

    # Format the model filenames based on input parameters
    formatted_train_split=$(echo $train_split | tr -d '.')
    formatted_lr=$(echo $lr | tr -d '.')
    formatted_momentum=$(echo $momentum | tr -d '.')

    if $include_model_classic; then
        model_classic="${subfolder}/mnist_classic_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
    fi
    model_sync="${subfolder}/mnist_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
    model_async="${subfolder}/mnist_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"

    if $include_model_classic; then
        python3 nn_train.py --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed --subfolder $subfolder
        sleep 0.1
        echo
        python3 test_model.py $model_classic $test_model_flags --subfolder $subfolder
        sleep 0.1
        echo
    fi

    python3 dnn_sync_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed --subfolder $subfolder
    sleep 0.1
    echo
    python3 test_model.py $model_sync $test_model_flags --subfolder $subfolder
    sleep 0.1
    echo

    python3 dnn_async_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed --subfolder $subfolder
    sleep 0.1
    echo
    python3 test_model.py $model_async $test_model_flags --subfolder $subfolder
    sleep 0.1
    echo
  done
done