#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import functools
import collections
import opt_einsum as oe

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.index    as tid

import tadpole.tensor.core        as core
import tadpole.tensor.interaction as tni
import tadpole.tensor.reindexing  as reidx


from tadpole.tensor.types import (
   Engine,
   IndexProduct,
)


from tadpole.tensor.engine import (
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
###  The logic of index contraction: figures out the output indices,        ###
###  generates einsum equations from input and output indices.              ###
###                                                                         ###
###############################################################################


# --- Symbol generator (generates symbols corresponding to Index objects) --- #

class Symbols:

   def __init__(self, symbolfun=None):

       if symbolfun is None:
          symbolfun = unicode_symbol

       self._symbolfun = symbolfun


   @util.cacheable
   def _map(self):

       next_symbol = map(self._symbolfun, itertools.count()).__next__

       return collections.defaultdict(next_symbol)


   def next(self, ind):

       return self._map()[ind]




# --- Unicode symbol function ----------------------------------------------- #

@functools.lru_cache(2**12)
def unicode_symbol(ind):

    return oe.get_symbol(ind)




# --- Helper: lru cache for indices (does not throw if __eq__ fails)  ------- # 

def lru_cache_indices(*cache_args, **cache_kwargs):

    lru_cache_wrap = functools.lru_cache(*cache_args, **cache_kwargs) 

    def wrap(fun):
        
        def _wrap(*args, **kwargs):

            try:
               return lru_cache_wrap(fun)(*args, **kwargs)
            except (ValueError, AssertionError):
               return fun(*args, **kwargs)

        return _wrap

    return wrap




# --- Create einsum equation from input and output indices ------------------ #

@lru_cache_indices(2**12)
def make_equation(input_inds, output_inds): 

    symbols = Symbols(unicode_symbol)

    def tosymbols(inds):
        return "".join(map(symbols.next, inds))

    inputs = tuple(tosymbols(inds) for inds in input_inds)
    output = tosymbols(output_inds)

    return ",".join(inputs) + f"->{output}"




# --- Fixed index product --------------------------------------------------- #

class IndexProductFixed(IndexProduct):

  def __init__(self, inds):

      self._inds = inds


  def __call__(self, inds):

      inds = Indices(*util.unique(util.concat(inds))) 
  
      return iter(inds.map(*self._inds))




# --- Pairwise index product ------------------------------------------------ #

class IndexProductPairwise(IndexProduct): 

   def __call__(self, inds):

       for ind, freq in util.frequencies(util.concat(inds)).items():

           if freq > 2:
              raise ValueError(
                 f"{type(self).__name__}: "
                 f"Index {ind} appears more than twice! All input indices: "
                 f"{inds}. If you wish to perform a hyper-index contraction, "
                 f"please specify the output indices explicitly."
              )

           if freq == 1:
              yield ind 




# --- Trace index product --------------------------------------------------- #

class IndexProductTrace(IndexProduct):

  def __call__(self, inds):

      output_inds = Indices(*inds[0])

      for eye_inds in inds[1:]:
          output_inds = output_inds ^ Indices(*eye_inds)  

      return iter(output_inds)




###############################################################################
###                                                                         ###
###  Tensor contraction engine                                              ###
###                                                                         ###
###############################################################################


# --- Tensor contraction factory -------------------------------------------- #

def tensor_contract(*xs, product=None):

    if product is not None and not isinstance(product, IndexProduct): 
       product = IndexProductFixed(product)

    engine = EngineContract(product)

    for x in xs:
        engine = x.pluginto(engine)

    return engine.operator()




# --- Tensor contraction engine --------------------------------------------- #

class EngineContract(Engine): 

   def __init__(self, product=None, train=None):

       if product is None:
          product = IndexProductPairwise()

       if train is None:
          train = TrainTensorData()

       self._product = product
       self._train   = train


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._product, other._product)
          log.val(self._train,   other._train)

       return bool(log)


   def attach(self, data, inds):

       return self.__class__(self._product, self._train.attach(data, inds))


   def operator(self):

       return TensorContract(
                 tuple(self._train.data()), 
                 tuple(self._train.inds()), 
                 self._product
              )




###############################################################################
###                                                                         ###
###  Tensor dot engine                                                      ###
###                                                                         ###
###############################################################################


# --- Tensor dot factory ---------------------------------------------------- #

def tensor_dot(x, y):

    engine = EngineDot()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)

    return engine.operator()




# --- Tensor dot engine ----------------------------------------------------- #

class EngineDot(Engine): 

   def __init__(self, train=None):

       if train is None:
          train = TrainTensorData()

       self._train = train


   def attach(self, data, inds):

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       return TensorContract(
                 tuple(self._train.data()), 
                 tuple(self._train.inds()), 
                 IndexProductPairwise()
              )




###############################################################################
###                                                                         ###
###  Tensor contraction operator                                            ###
###  (includes contract, dot, and other operations)                         ###
###                                                                         ###
###############################################################################


# --- Tensor contraction operator ------------------------------------------- #

class TensorContract:

   # --- Construction --- #

   def __init__(self, data, inds, product): 

       self._data    = data
       self._inds    = inds
       self._product = product


   # --- Private helpers --- #

   def _equation(self):

       return make_equation(self._inds, self._output_inds())


   def _output_inds(self):

       return tuple(self._product(self._inds))


   def _output_tensor(self, data):

       return core.TensorGen(data, Indices(*self._output_inds()))


   # --- Main methods --- #

   def contract(self):

       data = ar.einsum(self._equation(), *self._data, optimize=True)

       return self._output_tensor(data)


   def dot(self):

       data = ar.dot(*self._data)
       
       return self._output_tensor(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to TensorContract methods           ###
###                                                                         ###
###############################################################################


# --- Contraction ----------------------------------------------------------- #

@ad.differentiable
def contract(*xs, product=None):

    op = tensor_contract(*xs, product=product)

    return op.contract() 




# --- Dot product ----------------------------------------------------------- #

def dot(x, y):

    return contract(x, y)




# --- Kronecker product ----------------------------------------------------- #

def kron(x, y, kronmap):

    return reidx.fuse(contract(x, y), kronmap)




# --- Trace ----------------------------------------------------------------- #

def trace(x, inds):

    lind, rinds = inds[0], inds[1:]
    eyes        = (core.space(x).eye(lind, rind) for rind in rinds)

    return contract(x, *eyes, product=IndexProductTrace())




