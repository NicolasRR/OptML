# Script for running all the experiments

# Run the asynchronous training
for d in 0.2 0.5 1 1.5
do
    python dnn_mnist_async.py --output ./results/dnn_mnist_async$d --d $d
    echo "Training finished with delay $d"
done

# for m in 0.5 0.8 0.9
# do
#     python dnn_mnist_async.py --output ./results/dnn_mnist_async$m --d $d
#     echo "Training finished with momentum $m"

# done 

# Generate plots

# for 
# do 


