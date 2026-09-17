+++
title = 'Scaling Laws'
weight = 4
+++
The next ingredient we want to look at is how to optimize our model size and number of tokens for training given a compute budget, typically measured in FLOPs (floating point operations). The token budget and model size both determine the achievable performance and how long it takes to get there. Suppose you trained a language model large enough to melt a small glacier and fund an energy company's next summer party, how would you get it right the first time?

A naive approach to training an LLM is to run a training loop and hope that the loss will go flat or go below a certain threshold. This approach has problems.

1. The loss function may never plateau or drop below a certain threshold under the chosen hyperparameters.

2. The loss function may do so eventually, but there is no telling when this will happen.

A smarter approach would seem to be hyperparameter optimisation (HPO). But LLMs have unique constraints, being extremely large and slow to train. You therefore essentially multiply the number of training runs. Even with early stopping, such as in Hyperband HPO, you have problems.

1. You might kill off "late bloomers", good candidates that seem comparatively weak. To mitigate this you would still have to train each model for an extremely long time before pruning.

2. You would require training several large models in parallel, basically multiplying the hardware budget.

A groundbreaking paper by J. Kaplan et al.[^kaplan2020] showed that large language models[^note_kaplan2020] exhibit scaling laws over many orders of magnitude, investigating performance scaling laws as function of training time, context length, dataset size, model size (in number of parameters) and compute budget. The paper showed that one can reasonably extrapolate how a model would perform if it were scaled up, or if more data were made available, and so forth. Quite significantly, many came to believe larger models would perform better than smaller models over more orders of magnitude still, a scaling hypothesis that many companies then had bet their bottom dollar on (though it is known that power laws cannot go on forever to reach 0 loss, as we shall see). This early scaling law paper became a kind of oracle for practitioners, offering a glimpse into the future of LLMs when more compute, data and context will be made available.

> [!Note]
> I recommend at least skimming this paper. In any case, a few interesting observations from the paper include:
> 1. Performance improves with increases in model size (number of parameters), dataset size, and compute.
> 3. Only a weak relation was found related to model shape, e.g., model depth and width.
> 4. Performance improves predictably when scaling up model size and dataset size concurrently. However, scaling up one or the other in isolation—as one expects—results in overfitting to the training set. Specifically, if you scale a model size by a factor of 8, you need to scale the data budget by roughly 5 to steer clear of overfitting.
> 5. Large models are more sample efficient, needing fewer samples to reach a given level of performance than a smaller model.
> 6. Roughly speaking, transfer to an unseen dataset with a different distribution incurs a constant penalty but otherwise improves according to the same underlying power law as for the original dataset.

Hoffman et al.[^hoffman2022] expand on the work of Kaplan, also looking at optimising the learning rate schedule.

They offer three approaches to finding optimal hyperparameters with various pros and cons. These are often dubbed "Chinchilla approaches".

1. Fix model sizes and vary number of training tokens

2. ISOFlops



> [!Note]
> Recent work has also moved past the Chinchilla equation. One problem with it is that it only considers the fixed budget of training a model, rather than the cost of online inference in downstream applications. Optimising for a training budget can result in large, expensive-to-run models.


In the coming chapt



In practice, the compute budget \(C\) is known beforehand and fixed.



With all that said, the IsoFLOPs approach introduced by Hoffman et al. is simple. We have three variables, \(C\), \(N\) and \(D\), related to each other by a pre-established, empirically discovered function \(f: f(C, N, D) = 0\), and an optimal loss \(L(C, N, D)\). We keep \(C\) fixed, as \(C\) is often known beforehand, and vary one parameter each, in our case model size \(N\), computing \(D\) from \(C\) and \(N\) and training the model to get \(L\).

Hoffman et al. observes a quadratic loss






Next mention [^hoffman2022].




[^kaplan2020]: Kaplan, Jared, et al. "Scaling Laws for Neural Language Models." arXiv preprint arXiv:2001.08361, 2020, arXiv:2001.08361.

[^note_kaplan2020]: Specifically decoder-only transformers for language generation.

[^hoffman2022]: Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." Advances in Neural Information Processing Systems, vol. 35, Neural Information Processing Systems Foundation, Inc., 2022, pp. 30016-30030.