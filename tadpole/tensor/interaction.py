#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import operator as operator_
from functools import reduce

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core       as core
import tadpole.tensor.reduction  as redu
import tadpole.tensor.reindexing as reidx

import tadpole.tensor.elemwise_unary  as unary
import tadpole.tensor.elemwise_binary as binary


from tadpole.tensor.types import (
   Engine,
)


from tadpole.tensor.engine import (
   EngineUnary,
   EngineElemwise,
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.index import (
   Index,
   IndexGen, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor interaction engine and operator                                 ###
###                                                                         ###
###############################################################################


# --- Tensor interaction factory -------------------------------------------- #

def tensor_interaction(*xs):

    engine = EngineInteraction()

    for x in xs:
        engine = x.pluginto(engine)

    return engine.operator()




# --- Tensor interaction engine --------------------------------------------- #

class EngineInteraction(Engine): 

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._train, other._train)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       return TensorInteraction(self._train.data(), self._train.inds())




# --- Tensor interaction operator ------------------------------------------- #  

class TensorInteraction:

   # --- Construction --- #

   def __init__(self, data, inds): 

       self._data = data
       self._inds = inds


   # --- Mutual index methods --- #

   def union_inds(self):

       out = reduce(operator_.or_, self._inds)

       return iter(util.unique(out)) 


   def overlap_inds(self):

       out = reduce(operator_.and_, self._inds)   

       return iter(util.unique(out)) 


   def complement_inds(self):

       out = reduce(operator_.xor, self._inds)

       return iter(util.unique(out)) 




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorInteraction methods        ###
###                                                                         ###
###############################################################################


# --- Mutual index methods -------------------------------------------------- #

@ad.nondifferentiable
def union_inds(*xs, **opts):

    op = tensor_interaction(*xs)
    return op.union_inds(**opts)


@ad.nondifferentiable
def overlap_inds(*xs, **opts):

    op = tensor_interaction(*xs)
    return op.overlap_inds(**opts)


@ad.nondifferentiable
def complement_inds(*xs, **opts):

    op = tensor_interaction(*xs)
    return op.complement_inds(**opts)




# --- Tensor matching ------------------------------------------------------- #

def match(x, target, **opts):

    return astype_like(reshape_like(x, target, **opts), target)




def astype_like(x, target):

    if unary.iscomplex(x) and not unary.iscomplex(target):
       x = unary.astype(unary.real(x), target.dtype) 

    if not unary.iscomplex(x) and unary.iscomplex(target):
       return unary.astype(x, target.dtype) 

    return x




def expand_like(x, target, inds=None):

    if inds is None:
       inds = tuple(complement_inds(target, x))

    out = reidx.expand(x, inds)
    out = reidx.transpose(out, *overlap_inds(target, out))

    return out




def reshape_like(x, target, keepinds=False): 

    if  not keepinds:
        target = reidx.squeeze(target) 

    for ind in complement_inds(x, target): 
        x = redu.sumover(x, (ind,))

    for ind in complement_inds(target, x):    
        x = reidx.expand(x, (ind,))

    return transpose_like(x, target)




def transpose_like(x, target):

    diff = tuple(complement_inds(x, target))
          
    if len(diff) == 0:
       return reidx.transpose(x, *union_inds(target))

    if len(diff) > 1:
       raise ValueError(
          f"transpose_like: "
          f"transposition is ill-defined because tensors {x} "
          f"and {target} have more than one index differing: {diff}."
       )

    diff,       = diff
    output_inds = (
                   ind if  ind in union_inds(x) else diff 
                       for ind in union_inds(target)
                  )

    return reidx.transpose(x, *output_inds)

 


