## Analysis of the delayed grad probl in Async SGD.pdf
- Downpour SGD, which minimizes worker idle time by allowing gradients computed on stale parameters to be sent to the parameter server.
- Direct usage of asynchronous SGD leads to added noise during training from stale gradients.
- Learning rate and batch size selection are the majorizing factors in whether delayed gradients significantly reduce test accuracy. Careful selection of learning rate and batch size, or use of adaptive learning rate
methods, is effective in minimizing the delayed gradient problem up to a large
($n = 257$) number of workers.
- Model parallelism, where a model is split across different workers, and data parallelism, where training data is sharded or distributed to worker copies.
- Due to the delayed application of gradients, ASGD suffers from accuracy degradation.
-  LeNet-5 model for the MNIST digit recognition task over 30 epochs,
using SGD with $m = 0.9$.
-  Simulate delayed gradient submissions in asynchronous SGD, we
create a wrapper around synchronous PyTorch optimizers which store gradients in a buffer and reapplies them after a constant delay.
- Test accuracy significantly degrades
at batch sizes larger than $256$ and $lr ≥ 10^{2.5}$
. We observe that regardless of delay amount, test
accuracy scales smoothly upward with lr, then sharply drops off. SGD with more delay is less
resilient to learning rate adjustment, with the test accuracy dropping off sooner. One plausible explanation for low test accuracy at high learning rates in delayed SGD is that gradients are more
variant near the start of training, so delayed updates add significant noise to the parameter search.
- Our experiments on Delayed SGD indicate that learning rate warmup only initially improves test accuracy but quickly falls off as soon as the full learning rate is applied. Learning rate
schedule likely has little effect in mitigating delayed gradient noise. We find that optimizers which
employ per-parameter adaptive learning rates, such as Adam (Kingma & Ba, 2014), increase resiliency to learning rate and batch size selections, achieving test accuracy comparable to baseline up
to 256 delay, in comparison to SGD, which cannot handle more than 32 delay.
## Asynchronous SGD with delay compensation.pdf
- With the fast development of deep learning, it has become common to learn big neural networks using massive training data. Asynchronous Stochastic Gradient Descent is widely adopted to fulfill this task for its efficiency
- **Asynchronous SGD are Asynchronous SGD known to suffer from the problem of delayed
gradients = local worker adds its gradient to the global model, the global model may have been updated by other workers and this gradient becomes “delayed”**
- synchronous SGD (SSGD), local workers compute the gradients over their own mini-batches of data, and then add the gradients to the global model. By using a barrier, these workers wait for each other, and will not continue their local training until the gradients from all the M workers have been added to the global model. It is clear that the training speed will be dragged by the slowest worker
- asynchronous SGD (Dean et al., 2012) has been adopted, with which no barrier is imposed, and each local worker continues its training process right after its gradient is added to the global model
- **Although ASGD can achieve faster speed due to no waiting overhead, it suffers from another problem which we call delayed gradient = before a worker wants to add its gradient $g(w_t)$ (calculated based on the model snapshot $w_t$) to the global model, several other workers may have already added their gradients and the global model has been updated to $w_{t+\tau}$ (here $\tau$ is called
the delay factor). Adding gradient of model wt to another model $w_{t+\tau}$ does not make a mathematical sense, and the training trajectory may suffer from unexpected turbulence.
This problem has been well known, and some researchers have analyzed its negative effect on the convergence speed**
- **According to the figure, local worker $m$ starts from $w_t$, the snapshot of the global
model at time $t$, calculates the local gradient $g(w_t)$, and then add this gradient back to the global model3. However, before this happens, some other $\tau$ workers may have already added their local gradients to the global model, the global model has been updated $\tau$ times and becomes $w_{t+\tau}$. The ASGD algorithm is blind to this situation, and simply adds the gradient $g(w_t)$ to the global model $w_{t+\tau}$ , as follows $w_{t+\tau+1} = w_{t+\tau} - \eta g(w_t)$**
- It is clear that the above update rule of ASGD is problematic (and inequivalent to that of sequential SGD): one actually adds a “delayed” gradient g(wt) to the current global model $w_{t+\tau}$ . In contrast, the correct way is to update the global model $w_{t+\tau}$ based on the gradient w.r.t. $w_{t+\tau}$

## Asynchronous Decentralized Parallel SGD.pdf
- Synchronous algorithms like AllReduce-SGD perform poorly in a heterogeneous environment
- asynchronous algorithms using a parameter server suffer from 
  - communication bottle-neck at parameter servers when workers are many
  - significantly worse convergence when the traffic to parameter server is congested
- Asynchronous is more robust to strugglers small workers compared to synchronous
- Asynchronous Parallel Stochastic Gradient Descent (Recht et al., 2011; Agarwal and Duchi, 2011 Feyzmahdavian et al., 2016; Paine et al., 2013) breaks the synchronization in S-PSGD by allowing workers to use stale weights to compute gradients. On nonconvex problems, when the staleness of the weights used is upper bounded, A-PSGD is proved to admit the same convergence rate as S-PSGD

## Asynchronous SGD for Transformers.pdf
- Asynchronous stochastic gradient descent (SGD) is attractive from a speed perspective because workers do not wait for synchronization
- Chen et al., 2016, 2018; Ott et al., 2018
- synchronous SGD, gradients are collected from all workers and summed before updating, equivalent to one large batch
- accumulation and waiting processes are absent in asynchronous SGD, where updates are applied immediately after they are computed by any processor. Since each update comes from one processor, the batch size per update in asynchronous SGD is smaller
- A stale gradient occurs when parameters have updated while a processor was computing its gradient. Staleness can be defined as the number of updates that occurred between the processor pulling parameters and pushing its gradient. Under the ideal case where every processor spends equal time to process a batch, asynchronous SGD with $N$ processors produces gradients with staleness $N-1$. Empirically, we can also expect an average staleness of $N-1$ with normally distributed computation time (Zhang et al., 2016).
