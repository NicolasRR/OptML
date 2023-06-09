#!/bin/bash

bash run_val.sh --dataset mnist --epoch 50
sleep 0.1
bash run_val.sh --dataset fashion_mnist --epoch 50
sleep 0.1