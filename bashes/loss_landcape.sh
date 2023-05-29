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

index=1
alt="alt_model"
# Iterate through the modified file names and run the Python script
for name in "${names[@]}"; do
  if ((index%5==0)) || [[ "$name" == *"$alt"* ]]; then
    python "$py/loss_landscape.py" "$name""_model.pt" "$name""_weights.npy" --subfolder "$training_directory/plots" 
  else
    python "$py/loss_landscape.py" "$name""_model.pt" "$name""_weights.npy" --subfolder "$training_directory/plots" &
  fi
  ((index++))
done

echo "finished creating the plots"

