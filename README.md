# tadpole

Tadpole provides a differentiable programming framework for tensor calculations. It handles tensor expressions using an indexing scheme based on the Einstein summation convention. The indexing can also be used to encode connectivity between tensors, which in turn is an essential ingredient for automating large-scale tensor contractions and building extended networks of tensors. 

Tadpole comes with an automatic differentiation engine, which supports backpropagation through arbitrary tensor expressions as well as forward-mode differentiation. It can also compute gradients and derivatives of any order. 

The original motivation for Tadpole stems from tensor network algorithms in computational quantum physics. But owing to its versatile approach, it can be used as a general purpose library for any problem that involves tensor expressions, mathematical functions containing gradients, and algorithms using gradient-based optimization. 

To get started with the package and become familiar with various examples of usage, see the examples directory.
