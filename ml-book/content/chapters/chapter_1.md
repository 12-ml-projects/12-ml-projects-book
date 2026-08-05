+++
date = '2026-07-27T19:29:54+02:00'
draft = false
title = 'Feedforward Neural Networks from Scratch'
+++

> "What I cannot create, I do not understand." - Richard Feynman

Machine learning fragments into many subdisciplines—there is no one place to start. A holistic discussion of the subject is difficult without abstract and mathematical writings unfit for this book. I won't try to define here what it means for a machine to "learn". Instead, we will start somewhere in the middle. The neural networks we'll construct are all supervised machine learning models, meaning that they learn from labeled data. These networks use learned patterns to make predictions on unseen data.

The simplest neural networks are a good place to start learning about machine learning—they are complex enough to require some thoughtful engineering, and flexible enough to discover complex patterns. They are readily visualised and in that way easy to intuit.

While we can write these networks up in a few lines of code with machine learning libraries, we will write them using just the Python standard library. After all, machine learning libraries leverage the same abstractions that we ourselves will build.

In this chapter we will look at the history of feedforward neural networks. We will explore the foundational concepts of supervised machine learning, including loss functions, gradient descent, regularisation, overfitting and underfitting, metrics, and more. We will also explore the inner workings of automatic differentiation and backpropagation.

## A Brief History
> "Ultimately, we will get real histories of Artificial Intelligence . . . , written with as much objectivity as the historians of science can muster. That time is certainly not yet". - Allen Newell in 1982

I cannot give a sound and complete account of artificial intelligence. The same can be said for neural networks. At best I can give one narrative of an ineffable history, in a way part of a neverending game of Chinese whisper. Before diving in, it is worth stating that the triumphs, trials and tribulations of neural network research encompass a vivid epic. The field has had quite a thorny past which I hope to preserve a little in this brief history. I also hope to convey some of the engineering wonder that were these early models.

<!-- The interest in neural networks, such as in Rosenblatt's perceptron model, was quite intense, even in the public eye. After all, the notion of artificial intelligence had an allure dating back to ancient myth, such as Talos and Pandora [Gods and Robots]. Furthermore, the concrete implementation of these algorithms, such as the first perceptron on the IBM 704, or ADALINE as custom hardware, mark impressive, costly feats of engineering. -->

The neural networks we will look at all have their roots in early research on cybernetics which held strong in the 1940s-1960s, as part of a movement that constitutes the first of three historical waves of development in so-called deep learning [Goodfellow 13]. At the time there was a strong interest in biological learning i.e., how learning happens in the real brain, in particular in the context of biological feedback and control[^n2010]. This field led to the development of rudimentary machine learning models.

Our story begins in the 40s, when McCulloch and Pitts formalised a class of biologically-inspired neurons[^mp1943] based on a formal calculus of nervous activity. Their system constructs neural networks in ways very similar to logical circuits. "The" McCulloch-Pitts neuron was essentially a network of nodes with binary inputs and outputs, corresponding to neurons, dendrites and axons respectively[^n2010]. One crucial limitation of their formalism was that it did not prescribe learning rules, and one therefore had to hand-code the neurons' circuitry and thresholds unaided.

Motivated by McCulloch and Pitts' work, Frank Rosenblatt's perceptron model[^r1958] of the late 1950's came with an automated learning rule[^fn_backprop_name], bringing us closer to modern machine learning with a model that could actually learn the weights from example data in order to learn to classify unseen data. The learning rule was essentially to move the weight vector to match the correct output. It is hard to overstate the sheer impact of this model; by the late 1950s there may have been as many as hundreds[^mp1969] (perhaps thousands[^o1996][^fn_thousands]) of research groups experimenting with perceptrons. In the public eye too, there was interest. The perceptron project funded by the US Office of Naval Research, became highly publicized following a 1958 press conference, in which Rosenblatt made several strong claims. A New York Times article the same year echoes that the perceptron to be the embryo of a computer that can "walk, talk, see, write, reproduce itself and be conscious of its existence"[^nyt1958]. As for Rosenblatt's own tall claims, they may have may have been originated in an otherwise modest intent for public support[^mp1969], though rhetorics of this kind would still go on to fuel persistent quarrels throughout the wider scientific community, in particular between Rosenblatt and Marvin Minsky.

In the meantime, and independently of Rosenblatt, the adaptive linear neuron (ADALINE) was another major development, bringing the world a simple neural network mathematically similar to the perceptron, as well as the physical device that implemented it. The training algorithm for ADALINE, known as the Widrow-Hoff least-mean-squares adaptive algorithm, was a specific instance of what is now recognised as stochastic gradient descent (SGD). SGD and its variations now constitute the main learning algorithms used in today's gradient-based deep learning.

Rosenblatt's single-layered perceptron, as well as ADALINE, were part of the family of what we now call linear models—geometrically speaking, these models can separate two types of data by drawing a line, plane, or higher-dimensional plane between them. They certainly could not learn to classify just any set of data. Indeed, while the perceptron seemed promising, it came to be known that they could not be trained on all patterns under certain conditions. One now-famous book by Minsky and Papert, Perceptrons[^mp1969], came to legitimise many qualms about perceptrons, formally showing that single-layered perceptrons could not learn certain families of functions, such as the binary XOR function. This refutation was a key opening move—albeit not the only one—in the closure of what has been dubbed the perceptron controversy[^fn_perceptrons] at the beginning of a neural network winter. According to the official history[^o1996] as accepted by the community at large, it was broadly interpreted, quite wrongly, that Minsky and Papert had effectively shown that progress in neural nets was not possible[^o1996]. Neural network research entered a recession, upstaged by symbolic systems—systems in which outputs emerge from the manipulation of symbols according to formal rules and strategies.[^fn_mp]

Historically, the wax and wane of neural network research has been repeatedly marked by a mix of socioeconomic and scientific factors. One factor which played a role in its revival in the 80s was the popularization of backpropagation, a leanrning algorithm developed by Rumelhart, Hinton and Williams[^fn_backprop].

[^mp1943]: McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The Bulletin of Mathematical Biophysics, 5(4), 115–133. doi:10.1007/BF02478259

[^n2010]: Nilsson, N. J. (2010). The Quest for Artificial Intelligence: A History of Ideas and Achievements. Cambridge University Press

[^o1996]: Olazaran, M. (1996). A Sociological Study of the Official History of the Perceptrons Controversy. Social Studies of Science, 26(3), 611–659. http://www.jstor.org/stable/285702

[^r1958]: Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386–408.

[^mp1969]: Minsky, M. L., & Papert, S. A. (1969). Perceptrons: An Introduction to computational geometry. MIT Press.

[^nyt1958]: New Navy device learns by doing. (1958, July 8). The New York Times, 25.

[^fn_backprop_name]: As a historical aside, Rosenblatt's learning rule carried the name of "back-propagating error-correction procedure" in the case of perceptrons with up to two trainable layers. The modern term "backpropagation" is a historical nod to this rule. See Rosenblatt, F. (1962). Principles of neurodynamics: Perceptrons and the theory of brain mechanisms. Spartan Books.

[^fn_backprop]: Paul Werbos had independently discovered an equivalent method for training neural networks dating back to 1974. Seppo Linnainmaa is also credited with his own independent discovery in 1970.

[^fn_mp]: It should be noted that Minsky and Papert, and the general majority, were well-aware that multi-layer perceptrons could represent the XOR function—any binary function for that matter. There was little doubt that deeper networks called for further investigation, but no one knew of powerful weight-adjusting rules for them. After all, Minsky and Papert's refutation predates the discovery of backpropagation. Their work also studies perceptrons from a cognitive, if not biological lens, characteristic of the time period. While today we might ask if "perceptrons are good approximators" or statistical learners, Minsky and Papert put perceptrons through the wringer of, for the most part, biological and gestalt-theoretical tests, for example in restricting the number of connections between input and output units, or in focusing on the ground-figure problem for a perceptronic "retina". I recommend Mikel Olazaran's 1996 paper for a more in-depth and somewhat contemporary look at the perceptrons controversy, which furthermore highlights the effects of social factors on science. Olazaran also gives his own brief history of neural networks.

[^fn_thousands]: The claim of thousands of alleged neural network research groups, reportedly from an interview in the autumn of 89, might very well just be an exaggeration by Minsky[^mp1969].

[^fn_perceptrons]: The arguments laid out in Perceptrons were already widely circulated in draft-form by the mid-1960s, and the controversy had already been in full swing long before its publication[^n2010].