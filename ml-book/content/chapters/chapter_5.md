+++
date = '2026-07-27T19:29:54+02:00'
weight = 5
draft = false
status = 'draft'
title = 'Language Model from Scratch'
+++
Transformer models were introduced in the 2017 paper *attention is all you need*[^vaswani2017] initially targeted at the problem of language. Not long after, it was realised they could be applied to a range of problems in a landmark paper by Dosovitskiy et. al., which applied transformers to the problem of image recognition[^dosovitskiy2021][^fn_vision_transformer].

It seems it has become somewhat of a rite of passage to replicate the transformer from the original paper. At the time of writing, the original transformer paper has been cited over 180,000 times, alongside in total 600 registered implementations on [paperswithcode](https://paperswithcode.com/) alone. Since then, the transformer architecture has spawned many lines of research focusing on modifications and tricks.

This comes as no surprise as transformers have, in the past years, been at the top of the list of ML models in common use. Their influence can be traced back to certain properties of the transformer architecture, namely that it has proven incredibly parallelizable and scalable, proving that very large language models (LLMs) can be made practical. It was observed in 2020 that transformer models appeared to have remarkable scaling properties; their performance seemed only to improve as a function of dataset size and model scale[^kaplan2020]. Bigger is better. This came to be called the scaling hypothesis. Further research observed that very large models were few-shot learners, meaning that they could rapidly pick up new, correct associations from little input[^brown2020]. There are now a handful of world-wide companies, such as for example OpenAI, Anthropic and Mistral AI, built around gargantuan proprietary LLMs, whose flagship products are such LLMs—albeit more sophisticated and "agentic"—and whose entire business model rests on, roughly speaking, generating tokens. This confluence of factors has given rise to a new era of agentic AI, wherein communication among LLMs, programmatic tools, and other types of models, are powering the modern AI personal and code assistants common to every household.

## A Dive into PyTorch
It would be too much to build a transformer right away without knowing the basics of PyTorch. We have seen in chapter 1 some of the abstractions powering modern ML frameworks, and in chapter 2 how to use scikit-learn and keras. This is already a strong foundation for understanding PyTorch.

PyTorch is an ML framework focused on deep learning.

## How do Transformers Work

## 

[^dosovitskiy2021]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. Proceedings of the International Conference on Learning Representations (ICLR)

[^vaswani2017]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems (Vol. 30). Curran Associates, Inc.

[^kaplan2020]: Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., … Amodei, D. (2020). Scaling Laws for Neural Language Models. arXiv [Cs.LG]. Retrieved from [http://arxiv.org/abs/2001.08361](http://arxiv.org/abs/2001.08361)

[^brown2020]: Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., … Amodei, D. (2020). Language Models are Few-Shot Learners. arXiv [Cs.CL]. Retrieved from [http://arxiv.org/abs/2005.14165](http://arxiv.org/abs/2005.14165)

[^fn_vision_transformer]: My strong recommendation to see an implementation of this paper is [chapter 8](https://www.learnpytorch.io/08_pytorch_paper_replicating/) of [learnpytorch.io](https://learnpytorch.io/)