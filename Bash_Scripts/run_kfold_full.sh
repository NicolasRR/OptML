#!/bin/bash

script_dir=$(dirname "$0")
project_dir=$(realpath "$script_dir/..")
run_kfold_sh="$project_dir/Bash_Scripts/run_kfold.sh"

bash $run_kfold_sh
sleep 0.1
bash $run_kfold_sh --alr