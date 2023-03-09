#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util             as util
import tadpole.autodiff         as ad
import tadpole.array.operations as op




# --- Non-differentiable array operations ----------------------------------- #
 
dtype = ad.nondifferentiable(op.dtype)
size  = ad.nondifferentiable(op.size)
ndim  = ad.nondifferentiable(op.ndim)
shape = ad.nondifferentiable(op.shape)

allequal = ad.nondifferentiable(op.allequal)
allclose = ad.nondifferentiable(op.allclose)

put = ad.nondifferentiable(op.put)




# --- Differentiable array operations: unary -------------------------------- #

getitem = ad.differentiable(op.getitem)
reshape = ad.differentiable(op.reshape)

neg = ad.differentiable(op.neg)
sin = ad.differentiable(op.sin)
cos = ad.differentiable(op.cos)




# --- Gradient operations --------------------------------------------------- #

addgrads = ad.differentiable(op.addgrads)




# --- Differentiable array operations: binary ------------------------------- #

add = ad.differentiable(op.add)
sub = ad.differentiable(op.sub)
mul = ad.differentiable(op.mul)




# --- Differentiable array operations: nary --------------------------------- #

einsum = ad.differentiable(op.einsum)












