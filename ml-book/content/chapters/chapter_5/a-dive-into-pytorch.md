+++
title = 'A Dive into PyTorch'
weight = 1
+++
It would be too much to build a transformer right away without knowing the basics of PyTorch. We have seen in chapter 1 some of the abstractions powering modern ML frameworks, and in chapter 2 how to use scikit-learn and keras. This is already a strong foundation for understanding PyTorch.

PyTorch is an ML framework focused on deep learning. We have seen in chapter 1 how we can offload and abstract and concrete ML work and primitives to custom modules, creating a higher level API that hides many of these details. PyTorch in particular is an industry standard library of the sort for the purpose of deep learning, offering powerful abstractions for such things as importing data, neural networks primitives, automatic differentiation, loss functions, and more.

## Tensors
One of the basic primitives of PyTorch is the tensor[^tensor_note]. The tensor can be thought of as a higher-dimensional array of numbers, optimised for GPUs and TPUs (tensor processing units).

### Creating Tensors
Scalars can be written as
```python
scalar = torch.tensor(7.25)
scalar
# tensor(7.2500)
```
And a scalar is really just a 0-dimensional tensor.

Higher dimensional tensors can similarly be constructed from lists, as well as from numpy arrays.

```python
vector = torch.tensor([1, 0, 0])
# tensor([1, 0, 0])

matrix = torch.tensor([[1, 0, 0], [1, 3, 0], [0, 0, 2]])
matrix
# tensor([[1, 0, 0],
#         [1, 3, 0],
#         [0, 0, 2]])

tensor_3d = torch.tensor(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        [
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        [
            [19, 20, 21],
            [22, 23, 24],
            [25, 26, 27],
        ],
    ]
)
```

A tensor has a dimension to it. The dimension, also the tensor's length, tells you the number of indices. To understand the range of numbers that an index indexes, you need to use the `shape` property. The shape of a tensor tells you the "length", so to speak, of the tensor along each index. The size of a tensor, however, is the number of entries i.e., the product of the entries in its `shape`.

```python
scalar.ndim     # 0
vector.ndim     # 1
tensor_2d.ndim  # 2

tensor_3d.ndim  # 3
tensor_3d.shape # torch.Size([3, 3, 3])
tensor_3d.size  # 27
```

A tensor's entries all have the same data type. The default data type is `torch.float32` for representing floats and `torch.int64` for ints. However, these can be chosen during initialization.

```python
matrix_64f = torch.tensor([[1.0, 0], [0, 1.0]], dtype=torch.float64)
matrix_64f.dtype
# torch.float64
```


Tensors are "Pythonic" in the sense that many operators have been overridden for our convenience, giving a more idiomatic API. The dopest such operation I think is slicing. You can slice, dice and divvy up your tensor in numerous ways.

```python
tensor_3d[0,1,2]
# tensor(6)

# fix the index along the first axis to its size - 1, and keep all other elements
tensor_3d[-1,:,:]
# tensor([[19, 20, 21],
#         [22, 23, 24],
#         [25, 26, 27]])

assert tensor_3d[-1,...].equal(tensor_3d[-1,:,:])   # True
```

> [!NOTE]
> I recommend slicing in one fell swoop, without chaining—avoid writing, for example, `tensor_3d[0][1:2]`. `tensor_3d[0:1][2]` is not the same as the `tensor_3d[0:1,2]`. This is because a slice  preverses the dimension of the tensor, so `x = tensor_3d[0:1]` then `x[2]` is taking the second index on the first element of a 3D tensor, whose first dimension has size `1`.

We can also use Python ellipses to fix parts of the slice and select everything else.

```python
tensor_3d[...,1]
# tensor([[2, 5, 8],
#        [11, 14, 17],
#        [20, 23, 26]])
```
Here we are fixing the innermost index (index 2) to 1, and selecting every other combination of indices for the rest (index 0 and 1).

And can we reason intuitively about 3D tensors slices? In fact yes! You can lay out your tensor as a cube in 3D space. You can think of slicing up this cube, first from the side, then from above, then from the front, giving a cuboid corresponding to the resulting tensor.

___ INSERT IMAGE ___









[^tensor_note]: not to be confused with tensors from e.g., physics courses or quantum computing—ML tensors are not, by default, interpreted as coordinate representations of a mathematical tensor like a bilinear form. Tensors in machine learning are really just multidimensional arrays of numbers, nothing more, constructed with the intention of performing operations quite similar to those on 2D arrays—multilinear algebra, that is. For example, multiplication of tensors can be achieved by multiplying and summing entries over matching indices, carrying untouched indices along. If you're familiar with Einstein notation, it's quite close to that, and in fact Einstein notation gives a very convenient way to represent transformations on tensors.
