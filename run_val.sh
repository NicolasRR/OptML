#!/bin/bash

#dos2unix compare.sh
#bash compare.sh
#bash compare.sh > compare_result.txt

include_model_sgd=false # include classic sgd to compare with sync and async
include_classification_report=false # include labels classification report
notebook=false # use notebook for plots
alr=false # Adaptive Learning Rate
lrs="" # Learning rate scheduler
alt_model=false
train_split=1.0
pics=true
epoch=2
world_size=3
lr=0.01
momentum=0.0
batch_size=32
dataset="mnist"

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_sgd) include_model_sgd=true; shift ;;
    --classification_report) include_classification_report=true; shift ;;
    --notebook) notebook=true; shift;;
    --epoch) epoch="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --alr) alr=true; shift ;;
    --lrs) lrs="$2"; shift 2 ;;
    --alt_model) alt_model=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Format the model filenames based on input parameters
formatted_train_split=$(echo $train_split | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')

if $alt_model; then
    model_classic="${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_alt_model.pt"
else
    model_classic="${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
fi

if $include_model_sgd; then
    if $alt_model; then
        model_sync="${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}_alt_model.pt"
    else
        model_sync="${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
    fi
fi

training_flags="--val"
if $alr; then
    training_flags+=" --alr"
fi
if $alt_model; then
    training_flags+=" --alt_model"
fi
if [ ! -z "$lrs" ]; then
    training_flags+=" --lrs $lrs"
fi

python3 nn_train.py --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed $training_flags
sleep 0.1
echo
if $include_model_sgd; then
    python3 dnn_sync_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed $training_flags
    sleep 0.1
    echo
fi

if $notebook; then 
    papermill_command="papermill Compare_and_Plot.ipynb Compare_and_Plot_out.ipynb"
    papermill_command+=" -p model_classic $model_classic"
    if $include_model_sgd; then
        papermill_command+=" -p model_sync $model_sync"
        
    fi
    if $include_classification_report; then
        papermill_command+=" -p include_classification_report $include_classification_report"
    fi
    papermill_command+=" -p include_pics $pics"

    eval $papermill_command
else
    test_model_flags="--training_time --pics"
    if $include_classification_report; then
        test_model_flags+=" --classification_report"
    fi
    python3 test_model.py $model_classic $test_model_flags
    sleep 0.1
    echo
    if $include_model_sgd; then
        python3 test_model.py $model_sync $test_model_flags
        sleep 0.1
        echo
    fi
fi
