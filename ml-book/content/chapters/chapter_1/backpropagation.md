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
> Oftentimes, we will hear practitioners talk about "the gradient of the loss function", which is really saying something very specific—the gradient of the loss as a function of the weights, keeping the inputs and outputs fixed. Abstractly, the whole pipeline relates inputs, weights and outputs to some real-valued loss. \(F(x, w, y) = L(M(x | w), y)\), and it is with respect to this function that we take partial derivatives of the weights, but not the inputs and outputs.
>
> Multiplying out your weights and taking a gradient like this directly is unwise though, since it would be computationally inefficient. The fact that we know the parts making up the network—activation functions, matrix multiplications, convolutions—in the first place is precisely what allows us to calculate over these parts individually, efficiently, and combine them into a final answer. We shall see that automatic differentiation, the most sophisticated form described here, is in principle just the chain rule combined with caching and clever recombination.

## Numerical Differentiation
In brief, numerical differentiation approximates the derivative of a function by sampling two nearby points. The derivative of a real-valued function is[^note_finite_diffs]

\[
f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}
\]

so we stop short of the limit and take a small but finite \(h\):

\[
f'(x) \approx \frac{f(x+h)-f(x)}{h}
\]

For many reasonable functions, the difference \(h\) does not have to be extremely close to \(0\) to get an acceptable approximation of \(f'(x)\)[^footnote_small_diff]

One way, then, to get the gradient \( \nabla_{\textbf{w}} L \) of the loss is to approximate it. But note that this gradient is a vector of \(n_{w}\) partial derivatives, where \(n_{w}\) is the number of parameters, and each of them needs its own nudged forward pass—\(n_{w} + 1\) evaluations of the network in total, or twice that for central differences. For a model with millions of parameters, a single training step would cost millions of forward passes. Numerical differentiation stays useful as a way to check a gradient obtained by other means, on a handful of coordinates, but it cannot train a network.

## Computational Graphs and Differentiation
Let us focus on a test function, \(e(x, y): \mathbb{R}^{2} \rightarrow \mathbb{R}\) defined by a series of equations

\[
\begin{aligned}
a(x) = sin(x) \\
b(x) = a(x) + x \\
c(x, y) = x \times y \\
d(x, y) = c(x, y)^{2} \\
e(x, y) = d(x, y) + b(x)
\end{aligned}
\]

We can get \(f(x, y)\) by substitution. We pass in \(e(x)\) into \(f(x)\) to get \(f(x) = sin(x) + x\). We then 

The process of substituting in this way suggests a graph underlying the computation. We see that \(a\) depends on \(x\), \(b\) depends on \(a\) and \(x\) and so on. Of course, \(x\) and \(y\) do not depend on anything, and can be thought of as computational sources. The below table illustrates the dependencies.

| Value   | Depends On    |
| :---:   |    :----:     |
| \(x\)   | —             |
| \(y\)   | —             |
| \(a\)   | \(x\)         |
| \(b\)   | \(a\), \(x\)  |
| \(c\)   | \(x\), \(y\)  |
| \(d\)   | \(c\)         |
| \(e\)   | \(d\), \(f\)  |


-- INSERT IMAGE HERE --


This decomposition is chosen to expose a few underlying primitives. Each "step", so to speak, is an elementary function. \(a(x)\) applies the \(\sin\) function. But also \(b(x)\) can be thought of as addition, naemly as \(\text{add}(a(x), x)\), where \(add(a, b) = a + b\), and so on—in fact, \(c\) is doing multiplication over two values, \(d\) is just multipication (or exponentiation) over a value, and \(e\) is also addition. We call these primitive functions operators—in practice you can represent a large family of computations using a handful of operators.

The graph above is an example of a computation graph, and each node in it represents an application of an operation. Each edge represents a dependency. Computation graphs, together with the chain rule, form the basis of automatic differentiation. The above graph is also a directed acyclic graph, or DAG, meaning that each edge has a start and end, and there are no cycles in the graph. Why are there no cycles? Because if there were, there would be nodes whose value ultimately depended on the node knowing its value beforehand, a kind of "ask your mom, ask your dad"-type chain.

We see that when you use substitution to find \(f(x, y)\), you are essentially walking the graph from the inputs \(x\) and \(y\), and keeping a record of the expression evaluated at each node—you could also say each node "holds" the evaluation. If you traverse this graph in a topological order—that is, when you only visit a node if you have visited its dependencies—you can always make use of your record to find the next node. The final node will be \(f(x, y)\) and give you the answer.

### Differentiating without the Chain Rule
To differentiate \(f(x, y)\) with respect to its variables, one way is simply to find the closed form for \(f(x, y)\) in terms of \(x\) and \(y\). If we substitute via the process above we get \(f(x, y) = x^{2}y^{2} + x + sin(x)\). We can then take partial derivatives to get

\[
\begin{aligned}
\frac{\partial f}{\partial x} = 2xy^{2} + 1 + cos(x) \\
\frac{\partial f}{\partial y} = 2x^{2}y
\end{aligned}
\]

This works, but this requires us to substitute function expressions. Another problem is that this method of doing symbolic differentiation doesn't lend itself easily to a computer in the way that humans reason about it—in fact, it's not immediately clear how you would teach a computer to do this. Surely when you took the partial derivatives, you were thinking "well, this is so on and so multiplied by so and so"—we often employ the sum, product and reciprocal rule as given primitives. But a computer would just see a string of symbols and pluses and minuses, and so on.

- A computer's internal representation of the equation is not a string, but a computational DAG. Strings are secondary.
- A computer only needs the chain rule, and does not necessarily need to use the product, sum or reciprocal rules

With humans, it's the opposite.

### Differentiating with the Chain Rule
Let us look more closely at what a computer does. This time, we will rewrite the list of equations with the primitives made explicit.

\[
\begin{aligned}
a(x) = sin(x) \\
b(x) = \text{add}(a(x), x) \\
c(x, y) = \text{mul}(x, y) \\
d(x, y) = \text{mul}(c(x, y), c(x, y)) \\
e(x, y) = \text{add}(d(x, y), b(x))
\end{aligned}
\]

Let us find the partial derivative of \(e\) with respect to \(x\). The chain rule, written out exhaustively, tells us that
\[
\begin{aligned}
\frac{\partial b}{\partial x} = \frac{\partial \text{add}(a(x), x)}{\partial a} \frac{\partial a}{\partial x} + \frac{\partial \text{add}(a(x), x)}{\partial x} \frac{\partial x}{\partial x} \\
\frac{\partial c}{\partial x} = \frac{\partial \text{mul}(x, y)}{\partial x} \frac{\partial x}{\partial x} + \frac{\partial \text{mul}(x, y)}{\partial y} \frac{\partial y}{\partial x} \\
\frac{\partial d}{\partial x} = \frac{\partial \text{mul}(c(x, y), c(x, y))}{\partial c} \frac{\partial c}{\partial x} + \frac{\partial \text{mul}(c(x, y), c(x, y))}{\partial c} \frac{\partial c}{\partial x} \\
\frac{\partial e}{\partial x} = \frac{\partial \text{add}(c(x, y), b(x))}{\partial c} \frac{\partial c}{\partial x} + \frac{\partial \text{add}(c(x, y), b(x))}{\partial b} \frac{\partial b}{\partial x}
\end{aligned}
\]

Note that there is again an underlying computational graph. It is, in fact, the same as the one we saw earlier. But this time the nodes store the results of partial derivatives. For example, you can see that \(\frac{\partial e}{\partial x}\) depends on \(\frac{\partial d}{\partial x}\) and \(\frac{\partial b}{\partial x}\) in a similar way to how \(e\) itself depends on \(d\) and \(b\).

But what about the partial derivatives involving \(\text{mul}\) and \(\text{add}\)? These are primitives, and the computer will need to know the associated derivatives with respect to the inputs.

In code, this might look something like

```python
class Add(Operation):
    def apply(x: float, y: float) -> float:
        return x + y
    
    def partial(x: float, y: float, idx: int) -> float:
        return 1.0


class Mul(Operation):
    def apply(x: float, y: float) -> float:
        return x * y
    
    def partial(x: float, y: float, idx: int) -> float:
        if idx == 0:
            return y
        elif idx == 1:
            return x
        raise ValueError("idx must be `0` or `1`")
```

It is clear that the partial derivative with respect to \(x\) and \(y\) can be found programmatically by traversing the graph in topological order and keeping a record of partials.







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

[^footnote_small_diff]: Nor should you want an extremely small \(h\): shrinking \(h\) reduces the approximation error, but it also subtracts two nearly equal floating-point numbers.