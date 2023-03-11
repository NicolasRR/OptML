# Asynchronous SGD 

The idea is to have a server that coordinates the updates of the model' weights and send the mini-batches for computing the gradients in several GPUs.
This optimizer can reduce the training time. 
In You. et al, they developped asynchronous averaged SGD with can significantly reduce the training time compared to multiple passes SGD (which has high accuracy), somehow using a averaged SGD increases the accuracy and could replace multiple passes. In this paper they also highlighted the importance of **schedulers** for the learning rate. This could also prove an interesting research topic.

Possible projects ideas:
* As the project is not supposed to be focused on theory, we could implement the ASGD in python by controlling the memory and processes, we could meddle with the computer processors and simulate the GPU computation using the different CPU cores. We would have to implement parallel programming and memory locking. The idea would then be to asses it's regularizazion effect (we could first show that normal SGD acts as a regularizer, control, and then see if ASGD also does). We could also play with the different delays by locking the processor for a longer time and see how the global solution changes, i.e. we could lock the different processor the same time of each have a different locking time and see the results.
* How do learning rate schedulers impact training time of SGD? For this idea it would be better perhaps to use real-world data as we want a longer training? We could study the effect of a learning rate scheduler on SGD vs ASGD



# GANs




# General
It could be a good idea to generate dummy data for our model training as we would be able to know before hand the data distribution and we would know what are the optimal parameters (given the function)
If we are asked to analyze the landscape or shape of the local minima, we could start with a point P, create a pBall of radius r and see how the gradients changes in average in that pBall. By doing this we would assume that there is no special direction which produces a very high gradient change, in which cases can we justify this assumption?
