# Asynchronous SGD 

The idea is to have a server that coordinates the updates of the model' weights and send the mini-batches for computing the gradients in several GPUs.
This optimizer can reduce the training time. 
In You. et al, they developped asynchronous averaged SGD with can significantly reduce the training time compared to multiple passes SGD (which has high accuracy), somehow using a averaged SGD increases the accuracy and could replace multiple passes. In this paper they also highlighted the importance of **schedulers** for the learning rate. This could also prove an interesting research topic.

## Possible projects ideas:
* As the project is not supposed to be focused on theory, we could implement the ASGD in python by controlling the memory and processes, we could meddle with the computer processors and simulate the GPU computation using the different CPU cores. We would have to implement parallel programming and memory locking. The idea would then be to asses it's regularizazion effect (we could first show that normal SGD acts as a regularizer, control, and then see if ASGD also does). We could also play with the different delays by locking the processor for a longer time and see how the global solution changes, i.e. we could lock the different processor the same time of each have a different locking time and see the results.
* How do learning rate schedulers impact training time of SGD? For this idea it would be better perhaps to use real-world data as we want a longer training? We could study the effect of a learning rate scheduler on SGD vs ASGD
* Working again with SGD we could see the potential not only for implicit regularization (parameters not too big) but also whether this method allows better generalization than normal gradient descent or Adam

# GANs

Way too complex to be studied, I believe no one from the group has enough experience with this

# Distributed Stochastic gradient descent

As well as ASGD, there is another approach where instead of keeping track of the parameters using a single server to which all the workers communicate, DSGD allows workers to keep track of the parameters locally. In order to update the parameters you can use either a ring (you use the gradients from the rest of the workers) or a clique (only communicate with a small number of workers)

## Possible ideas
* We could design an experiment where we have several workers networks architecture, ones more connected than other and compare the local minima, generalization, regularization and computing time that this produces. Let's not compare ASGD and DSGS as we will be simulating the experiments (lack of clusters) and our implementations may not be optimal. 

# General
It could be a good idea to generate dummy data for our model training as we would be able to know before hand the data distribution and we would know what are the optimal parameters (given the function)
If we are asked to analyze the landscape or shape of the local minima, we could start with a point P, create a pBall of radius r and see how the gradients changes in average in that pBall. By doing this we would assume that there is no special direction which produces a very high gradient change, *in which cases can we justify this assumption?*
Furthermore, to asses the shape of the local minima we can also standardize the coefficients so they have the same unities and see how much the norm of the gradient changes with respect to each dimension. We could do this for SGD, Adam, AdaGrad, etc

*Would it be useful for the project to start with an introduction explaining why our topic may be interesting?*