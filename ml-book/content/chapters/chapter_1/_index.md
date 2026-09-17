+++
date = '2026-07-27T19:29:54+02:00'
weight = 1
draft = false
status = 'draft'
title = 'Feedforward Neural Networks from Scratch'
+++

> "What I cannot create, I do not understand." - Richard Feynman

Machine learning fragments into many subdisciplines—there is no one place to start. A holistic discussion of the subject is difficult without abstract and mathematical writings unfit for this book. I won't try to define here what it means for a machine to "learn". Instead, we will start somewhere in the middle. The neural networks we'll construct are all supervised machine learning models, meaning that they learn from labeled data. These networks use learned patterns to make predictions on unseen data.

The simplest neural networks are a good place to start learning about machine learning—they are complex enough to require some thoughtful engineering, and flexible enough to discover complex patterns. They are readily visualised and in that way easy to intuit.

While we can write these networks up in a few lines of code with machine learning libraries, we will write them using just the Python standard library. After all, machine learning libraries leverage the same abstractions that we ourselves will build.

In this chapter we will look at the history of feedforward neural networks. We will explore the foundational concepts of supervised machine learning, including loss functions, gradient descent, regularisation, overfitting and underfitting, metrics, and more. We will also explore the inner workings of automatic differentiation and backpropagation.
