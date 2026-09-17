+++
title = 'Backpropagation'
weight = 2
+++
Looking at the perceptron example, we might wish to add more layers. But we need to perform gradient descent over many layers of the network. For this we need to know some basic multivariate calculus. We will then look at the computational problem—once we know how to compute the gradient of a loss function 

## Differentiation
At an interior local minimum or maximum of a differentiable real-valued function, the derivative is \(0\). Note that the converse does not hold—a vanishing derivative can equally well attain saddle points or a plateau.

You'll recall derivatives from your school years and how they work. The calculus taught then was likely single-variable calculus. In machine learning, multivariable calculus is commonplace. While it is conceptually similar in many ways to single-variable calculus, the mathematics and the basic primitives are a bit more involved.

Roughly speaking, if you differentiate a single variable—also called univariate—function \(f\) at \(x'\), you get the slope of the tangent line at \(x'\): the one line through \((x', f(x'))\) that runs in the same direction the function does at that point. Another fundamental way to think about it is, if you move a tiny bit to the left or right, the derivative tells you how much you will move up or down.

It takes two dimensions—namely a sheet of grid paper—to plot a function of one variable. Similarly, to plot a function of two variables \(f(x, y)\) you'll need three dimensions, \(x\) going "in", \(y\) going "right" and \(z\) going "up". Now if we talk about "the" derivative of \(f\) at a point \((x', y')\) as a single number, we will hit a wall. I can lay a line against the surface there, but in fact I can lay down many, one for every direction I rotate it through, each with its own slope—this gives us hints that in fact two numbers are required to describe the derivative in its entirety, in this case the rotation and inclination of the line on the plane. The object that generalises the tangent line is not a line, but a plane:

\[
p(x, y) = f(x', y') + a(x - x') + b(y - y')
\]

\(a\) and \(b\) are undetermined. The trick to find them is to hold one variable still, which drops us back into the single-variable case. We define the restriction \(g(y) = f(x', y)\) as the function obtained by freezing \(x\) at \(x'\). Note that this is really just a single variable function in the variable \(y\), with \(x\) now acting like a constant. Restricting the plane in the same way leaves \(p(x', y) = f(x', y') + b(y - y')\), a line in \(y\) of slope \(b\). For the plane to approximate \(f\) along this slice, the two slopes must agree, giving \(b = g'(y')\). Freezing \(y\) at \(y'\) instead leaves \(k(x) = f(x, y')\) and, by the same argument, \(a = k'(x')\)[^footnote_diff_derivation].

These two single-variable derivatives are what we call the partial derivatives of \(f\):

\[
\begin{aligned}
\frac{\partial f}{\partial x}(x, y) &= \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h} \\
\frac{\partial f}{\partial y}(x, y) &= \lim_{h \to 0} \frac{f(x, y + h) - f(x, y)}{h}
\end{aligned}
\]

It is readily seen that \(a = \frac{\partial f}{\partial x}(x', y')\) and \(b = \frac{\partial f}{\partial y}(x', y')\), so the plane is

\[
p(x, y) = f(x', y') + \frac{\partial f}{\partial x}(x', y')\,(x - x') + \frac{\partial f}{\partial y}(x', y')\,(y - y')
\]

Collecting the partial derivatives into a vector defines the gradient of \(f\),

\[
\nabla f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
\]

which is the "slope" of that plane, so to speak. Using the dot product, the plane compresses to

\[
p(\textbf{x}) = f(\textbf{x}') + \nabla f(\textbf{x}') \cdot (\textbf{x} - \textbf{x}')
\]

This readily generalizes the univariate case, as you can check by just using one variable in place of two: the dot product becomes ordinary multiplication and we recover the tangent line \(f(x') + f'(x')(x - x')\).

> [!Note]
> In practice, partial derivatives aren't hard to work with. You just remind yourself that the other variables are constants when taking a partial derivative.

In machine learning we don't typically predict a single real number, nor do we have a single parameter to tune. While it is true that the loss function spits out a single number, between your inputs and the loss function are a series of functions that take high-dimensional data to high-dimensional data. Our machine learning models can be, generally speaking, be broken up into a series of functions of the form \(f:\mathbb{R}^{n} \rightarrow \mathbb{R}^{m}\). The model as a whole can also be interpreted as one such function where the domain represents the set of all weights (including inputs, held fixed), and the codomain represents the set of all outputs.

> [!Note]
> Oftentimes, we will hear practitioners talk about "the gradient of the loss function", which is really saying something very specific—the gradient of the loss as a function of the weights, keeping the inputs and outputs fixed. Abstractly, the whole pipeline relates inputs, weights and outputs to some real-valued loss. \(F(x, w, y) = L(M(x : w), y)\) -- TODO: continue this paragraph -- Multiplying out your weights and taking a gradient like this directly is unwise though, since it would be computationally inefficient. The fact that we know the parts making up the network—activation functions, matrix multiplications, convolutions—in the first place is precisely what allows us to calculate over these parts individually, efficiently, and combine them into a final answer. We shall see that automatic differentiation, the most sophisticated form described here, is in principle just the chain rule combined with caching and clever recombination.





To reason


Each output coordinate \(f_{i}\) is a real-valued function of \(n\) variables, so it has a gradient of its own. Stacking those \(m\) gradients as rows gives the Jacobian

\[
J_{f} =
\begin{bmatrix}
\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{1}}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{m}}{\partial x_{1}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}}
\end{bmatrix}
=
\begin{bmatrix}
\nabla f_{1} \\
\vdots \\
\nabla f_{m}
\end{bmatrix}
\]

an \(m \times n\) matrix holding every partial derivative of every output with respect to every input.

Read it back through the intuition of small changes. Row \(i\) is the gradient of output \(i\), and answers "if I nudge the inputs, how does this one output respond?". Column \(j\) collects the partial derivatives with respect to \(x_{j}\), and answers the complementary question, "if I nudge this one input, how do all of the outputs respond?". Nudge every input at once by a small displacement \(\boldsymbol{\delta}\) and the two readings combine into

\[
f(\textbf{x} + \boldsymbol{\delta}) \approx f(\textbf{x}) + J_{f}(\textbf{x})\,\boldsymbol{\delta}
\]

which is the tangent plane again, with a matrix where the gradient used to be. The Jacobian, in other words, is the local linear approximation of \(f\).

Two cases are worth naming. When \(m = 1\) the Jacobian is a single row and is nothing other than the gradient; this is the case of a loss function, which swallows many parameters and returns one number. When \(n = 1\) it is a single column, describing a curve through \(\mathbb{R}^{m}\).

<!-- TODO: image of the Jacobian as a stack of gradients -->

There are a few methods of differentiating a function composed of other functions, and all rely on the chain rule. In the language above the chain rule is a matrix product: the Jacobian of \(g \circ f\) is \(J_{g}J_{f}\), each factor evaluated at the appropriate point. Backpropagation, as we will see, comes down to a careful choice of the order in which such a chain of matrices is multiplied.

## Numerical Differentiation
In brief, numerical differentiation approximates the derivative of a function by sampling two nearby points. The derivative of a real-valued function is[^note_finite_diffs]

\[
f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}
\]

so we stop short of the limit and take a small but finite \(h\):

\[
f'(x) \approx \frac{f(x+h)-f(x)}{h}
\]

For many reasonable functions, the difference \(h\) does not have to be extremely close to \(0\) to get an acceptable approximation of \(f'(x)\). Nor should it be: shrinking \(h\) reduces the approximation error, but it also subtracts two nearly equal floating-point numbers, and the resulting cancellation eats precisely the digits we are trying to measure. In double precision a forward difference is at its best around \(h \approx 10^{-8}\). A similar formula exists in the multivariate case, where you would be looking for partial derivatives, nudging one coordinate at a time and leaving the others alone.

One way, then, to get the gradient \( \nabla_{\textbf{w}} L \) of the loss is to approximate it. But note that this gradient is a vector of \(n_{w}\) partial derivatives, where \(n_{w}\) is the number of parameters, and each of them needs its own nudged forward pass—\(n_{w} + 1\) evaluations of the network in total, or twice that for central differences. For a model with millions of parameters, a single training step would cost millions of forward passes. Numerical differentiation stays useful as a way to check a gradient obtained by other means, on a handful of coordinates, but it cannot train a network.

## Symbolic Differentiation
Another way to do so is symbolically. There is a matrix calculus which can be exploited. Take for example the following neural network, defined mathematically:

\[
\begin{aligned}
\textbf{x} &\in \mathbb{R}^{n} \\
\textbf{h} &= \sigma(W_{0}\textbf{x} + \textbf{b}_{0}) \\
y &= W_{1}\textbf{h} + b_{1}
\end{aligned}
\]

with \(W_{0} \in \mathbb{R}^{k \times n}\), \(\textbf{b}_{0} \in \mathbb{R}^{k}\), \(W_{1} \in \mathbb{R}^{1 \times k}\) and \(b_{1} \in \mathbb{R}\), so that the network maps an input vector to a single number. Here \(\sigma\) is understood as applying a single-variable function element-wise.

To perform gradient descent we need the gradient not of \(y\) but of a loss. Measuring the prediction against a target \(t\) by the squared error \(L = (y - t)^{2}\), the quantities we are after are

\[
\frac{\partial L}{\partial W_{1}}, \quad
\frac{\partial L}{\partial b_{1}}, \quad
\frac{\partial L}{\partial W_{0}}, \quad
\frac{\partial L}{\partial \textbf{b}_{0}}
\]

and each one is, by the chain rule, a product of the Jacobians of the steps lying between that parameter and the loss.

[^note_finite_diffs]: there is also a much better approximation with \(
f'(x) = \lim_{h \to 0} \frac{f(x+h/2)-f(x - h/2)}{h}\) whose error term is \(O(h^{2})\) rather than \(O(h)\). This is also called a central finite difference, as opposed to a forward difference.

[^footnote_diff_derivation]: You might object, though, that we have constrained the plane along the x and y axis. But this doesn't tell us the plane meets the function as desired along other lines, like \(x=y\). In fact it does, and showing this requires exploiting the linearity of the derivative.
