#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core as core


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.engine import (
   EngineUnary,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)




@ad.differentiable
def flip(x, flipinds=None):

    def fun(data, inds): 

        axes = None
        if flipinds is not None:
           axes = inds.axes(*flipinds)

        outdata = ar.flip(data, axes)
        return core.Tensor(outdata, inds)

    return fn.Args(x).pluginto(fn.Transform(fun))




@ad.nondifferentiable
@typecast_unary
def iscomplex(x):

    def fun(data):
        return ar.iscomplex(data)

    return fn.Args(x).pluginto(fn.Extract(fun))  






def typematch(x, target):

    if iscomplex(target) and not iscomplex(target):
       return real(target)

    if not iscomplex(target) and iscomplex(target):
       return target + 0j

    return target



def shapematch(x, target, keepinds=False): # TODO can add keepinds to sumover too!

    if not keepinds:
       target = squeeze(target) # removes axes with dim=1 in target, so that they all get summed over

    for ind in x.space() ^ target.space():
        x = sumover(x, ind)

    for ind in target.space() ^ x.space(): 
        x = extend(x, ind)

    return x



def match(x, target, **opts)

    return typematch(shapematch(x, target, **opts), target) 



def unreduce_grad(x, target, inds=None): # TODO this seems exclusively for grads

    def fun(g):

        g1 = extend(g, target, inds)
        x1 = extend(x, target, inds)

        mask = core.isequal(target, x1)

        return g1 * mask / sumover(mask, inds)

    return fun






# --- Tensor member methods: basic functionality ---------------------------- #

@ad.nondifferentiable
def copy(x, **opts):

    return x.copy(**opts)


@ad.nondifferentiable
def todense(x):

    return x.todense()


@ad.nondifferentiable
def withdata(x, data):

    return x.withdata(data)


@ad.nondifferentiable
def space(x):

    return x.space()


@ad.nondifferentiable
def item(x, *pos):

    return x.item(*pos)




# --- Tensor member methods: properties ------------------------------------- #

@ad.nondifferentiable
def dtype(x):

    return x.dtype


@ad.nondifferentiable
def size(x):

    return x.size


@ad.nondifferentiable
def ndim(x):

    return x.ndim


@ad.nondifferentiable
def shape(x):

    return x.shape




# --- Tensor methods: element access ---------------------------------------- #

@ad.differentiable
def getitem(x, pos):

    def fun(u):
        return u[pos] 

    return fn.Args(x).pluginto(fn.Transform(fun))





# --- Tensor methods: gradient accumulation --------------------------------- #

@ad.differentiable
@typecast_binary
def addgrads(x, y):

    return y.addto(x)















