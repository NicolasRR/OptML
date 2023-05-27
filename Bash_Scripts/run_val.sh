#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
nn_train_py="$project_dir/nn_train.py"
test_model_py="$project_dir/test_model.py"
subfolder="$project_dir/Validation_results"


include_classification_report=true # include labels classification report
notebook=false # use notebook for plots
train_split=1.0
pics=true
epochs=6
lr=0.01
momentum=0.9
batch_size=32
dataset="mnist"

# Parse command-line arguments using flags
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model_classic) include_model_classic=true; shift ;;
    --classification_report) include_classification_report=true; shift ;;
    --notebook) notebook=true; shift;;
    --epochs) epochs="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --momentum) momentum="$2"; shift 2 ;;
    --batch_size) batch_size="$2"; shift 2 ;;
    --dataset) dataset="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Format the model filenames based on input parameters
formatted_train_split=$(printf "%.1f\n" $(echo "$train_split * 10" | bc) | tr -d '.')
formatted_lr=$(echo $lr | tr -d '.')
formatted_momentum=$(echo $momentum | tr -d '.')


model_classic="${dataset}_classic_0_${formatted_train_split}_${formatted_lr}_${formatted_momentum}_${batch_size}_${epochs}.pt"

python3 $nn_train_py --train_split $train_split --epochs $epochs --dataset $dataset --lr $lr --momentum $momentum --batch_size $batch_size --seed --val --saves_per_epoch 3 --subfolder $subfolder
sleep 0.1
echo

if $notebook; then 
    papermill_command="papermill Compare_and_Plot.ipynb Compare_and_Plot_out.ipynb"
    papermill_command+=" -p model_classic $model_classic"
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
