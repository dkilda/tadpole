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

    engine = EngineInteraction(product)

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

       return reduce(operator_.and_, self._inds)


   def overlap_inds(self):

       return reduce(operator_.or_, self._inds)   


   def complement_inds(self):

       return reduce(operator_.xor_, self._inds)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorInteraction methods        ###
###                                                                         ###
###############################################################################


# --- Mutual index methods -------------------------------------------------- #

def union_inds(*xs):

    op = tensor_interaction(*xs)
    return op.union_inds()


def overlap_inds(*xs):

    op = tensor_interaction(*xs)
    return op.overlap_inds()


def complement_inds(*xs):

    op = tensor_interaction(*xs)
    return op.complement_inds()




# --- Tensor matching ------------------------------------------------------- #

def match_type(x, target):

    if unary.iscomplex(x) and not unary.iscomplex(target):
       return unary.real(x)

    if not unary.iscomplex(x) and unary.iscomplex(target):
       return x + 0j

    return x




def match_shape(x, target, keepinds=False): 

    if  not keepinds:
        target = reidx.squeeze(target) 

    for ind in complement_inds(x, target):   
        x = redu.sumover(x, ind)

    for ind in complement_inds(target, x):  
        x = reidx.expand(x, ind)

    return x




def match(x, target, **opts):

    return match_type(match_shape(x, target, **opts), target)




# --- Tensor matching (for gradients specifically) -------------------------- #

def expand_grad(x, target, inds=None): 

    if inds is None:
       inds = complement_inds(target, x)

    def fun(g):

        g1 = reidx.expand(g, inds)
        x1 = reidx.expand(x, inds)

        mask = binary.isequal(target, x1)

        return g1 * mask / redu.sumover(mask, inds)

    return fun




