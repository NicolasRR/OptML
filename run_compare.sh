#!/bin/bash

#dos2unix compare.sh
#bash compare.sh
#bash compare.sh > compare_result.txt

include_model_classic=false # include classic sgd to compare with sync and async
include_classification_report=false # include labels classification report
notebook=false # use notebook for plots
alr=false # Adaptive Learning Rate
lrs="" # Learning rate scheduler
saves_per_epoch="" # Saves per Epoch
val=false # Validation
delay=false # Delay
slow_worker_1=false # Slow Worker 1
pics=false
loss_landscape=false # load the weights perform PCA and compute the loss landscape
alt_model=false

train_split=0.2
epoch=2
world_size=3
lr=0.01
momentum=0.0
batch_size=32
dataset="mnist"

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_classic) include_model_classic=true; shift ;;
    --classification_report) include_classification_report=true; shift ;;
    --notebook) notebook=true; shift;;
    --train_split) train_split="$2"; shift 2 ;;
    --epoch) epoch="$2"; shift 2 ;;
    --world_size) world_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    --alr) alr=true; shift ;;
    --saves_per_epoch) saves_per_epoch="$2"; shift 2 ;;
    --lrs) lrs="$2"; shift 2 ;;
    --val) val=true; shift ;;
    --delay) delay=true; shift ;;
    --slow_worker_1) slow_worker_1=true; shift ;;
    --pics) pics=true; shift ;;
    --loss_landscape) loss_landscape=true; shift ;;
    --alt_model) alt_model=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Format the model filenames based on input parameters
formatted_train_split=$(echo $train_split | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')
if $include_model_classic; then
    model_classic="${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
fi
model_sync="${dataset}_sync_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"
model_async="${dataset}_async_${world_size}_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.pt"

training_flags=""
if $alr; then
    training_flags+=" --alr"
fi
if $alt_model; then
    training_flags+=" --alt_model"
fi
if [ ! -z "$saves_per_epoch" ]; then
    training_flags+=" --saves_per_epoch $saves_per_epoch"
fi
if [ ! -z "$lrs" ]; then
    training_flags+=" --lrs $lrs"
fi

# Special training flags for nn_train.py
training_flags_nn=$training_flags
if $val; then
    training_flags_nn+=" --val"
fi

# Special training flags for dnn_sync_train.py
training_flags_dnn_sync=$training_flags
if $val; then
    training_flags_dnn_sync+=" --val"
fi
if $delay; then
    training_flags_dnn_sync+=" --delay"
fi
if $slow_worker_1; then
    training_flags_dnn_sync+=" --slow_worker_1"
fi

# Special training flags for dnn_async_train.py
training_flags_dnn_async=$training_flags
if $delay; then
    training_flags_dnn_async+=" --delay"
fi
if $slow_worker_1; then
    training_flags_dnn_async+=" --slow_worker_1"
fi

if $include_model_classic; then
    python3 nn_train.py --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed $training_flags_nn
    sleep 0.1
    echo
fi
python3 dnn_sync_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed $training_flags_dnn_sync
sleep 0.1
echo
python3 dnn_async_train.py --world_size $world_size --train_split $train_split --epoch $epoch --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed $training_flags_dnn_async
sleep 0.1
echo

if $notebook; then 
    papermill_command="papermill Compare_and_Plot.ipynb Compare_and_Plot_out.ipynb"
    if $include_model_classic; then
        papermill_command+=" -p model_classic $model_classic"
    fi
    papermill_command+=" -p model_sync $model_sync -p model_async $model_async"
    if $include_classification_report; then
        papermill_command+=" -p include_classification_report $include_classification_report"
    fi
    if $pics; then
        papermill_command+=" -p include_pics $pics"
    fi
    eval $papermill_command
else
    test_model_flags="--training_time"
    if $include_classification_report; then
        test_model_flags+=" --classification_report"
    fi
    if $pics; then
        test_model_flags+=" --pics"
    fi
    if $include_model_classic; then
        python3 test_model.py $model_classic $test_model_flags
        sleep 0.1
        echo
    fi
    python3 test_model.py $model_sync $test_model_flags
    sleep 0.1
    echo
    python3 test_model.py $model_async $test_model_flags
    sleep 0.1
fi

if $loss_landscape; then
    if [ -z "$saves_per_epoch" ] || [ $saves_per_epoch -lt 1 ]; then
        echo "saves_per_epoch is not defined or less than 1, skipping loss landscape computation."
    else
        if $include_model_classic; then
            classic_weights="${dataset}_classic_weights_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.npy"
            python3 loss_landscape.py $model_classic $classic_weights 
        fi
        sync_weights="${dataset}_sync_weights_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.npy"   
        async_weights="${dataset}_async_weights_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epoch}.npy"
        python3 loss_landscape.py $model_sync $sync_weights 
        python3 loss_landscape.py $model_async $async_weights 
    fi
fi
