#!/bin/bash

py=".."  # Path to Python script

training_directory=$1

models=$(find "$training_directory" -type f -name "*.pt")

names=()
# Define the substring to remove
substring="_model.pt"

# Loop through each string and remove the substring
for str in $models; do
  name=${str//$substring/}  # Remove the substring from the file name
  names+=("$name")
done

# Iterate through the modified file names and run the Python script
for name in "${names[@]}"; do
  python3 "$py/test_model.py" "${name}_model.pt" "${name}_log.log" --classification_report --training_time --pics --subfolder "$training_directory/test_results"
done

