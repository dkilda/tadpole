#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import functools
import opt_einsum as oe

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core    as core
import tadpole.tensor.reindex as reindex


from tadpole.tensor.types import (
   Engine
)


from tadpole.tensor.engine import (
   TrainTensorData,
   TooManyArgsError,
)


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
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




# --- Create einsum equation from input and output indices ------------------ #

@functools.lru_cache(2**12)
def make_equation(input_inds, output_inds): 

    symbols = Symbols(unicode_symbol)

    def tosymbols(inds):
        return "".join(map(symbols.next, inds))

    inputs = tuple(tosymbols(inds)) for inds in input_inds)
    output = tosymbols(output_inds)

    return ",".join(inputs) + f"->{output}"




# --- Index product interface ----------------------------------------------- #

class Product(abc.ABC):

   @abc.abstractmethod
   def __call__(self, inds):
       pass
       



# --- Fixed index product --------------------------------------------------- #

class FixedProduct(Product):

  def __init__(self, inds):

      self._inds = inds


  def __call__(self, inds):

      inds = Indices(*util.concat(inds)) 
  
      return iter(inds.map(*self._inds))




# --- Pairwise index product ------------------------------------------------ #

class PairwiseProduct(Product): 

   def __call__(self, inds):

       for ind, freq in util.frequencies(inds).items():

           if freq > 2:
              raise ValueError(
                 f"{type(self).__name__}: "
                 f"Index {ind} appears more than twice! All input indices: "
                 f"{inds}. If you wish to perform a hyper-index contraction, "
                 f"please specify the output indices explicitly."
              )

           if freq == 1:
              yield ind 




###############################################################################
###                                                                         ###
###  Tensor contraction engine                                              ###
###                                                                         ###
###############################################################################


# --- Tensor contraction factory -------------------------------------------- #

def tensor_contract(*xs, product=None):

    if product is not None and not isinstance(product, Product): 
       product = FixedProduct(product)

    engine = EngineContract(product)

    for x in xs:
        engine = x.pluginto(engine)

    return engine.operator()




# --- Tensor contraction engine --------------------------------------------- #

class EngineContract(Engine): 

   def __init__(self, product=None, train=None):

       if product is None:
          product = PairwiseProduct()

       if train is None:
          train = TrainTensorData()

       self._product = product
       self._train   = train


   def attach(self, data, inds):

       return self.__class__(self._train.attach(data, inds))


   def operator(self):

       return TensorContract(
                 self._train.data(), self._train.inds(), self._product
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
                 self._train.data(), self._train.inds(), PairwiseProduct()
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

       return self._product(self._inds)


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

    return op.einsum(optimize) 




# --- Dot ------------------------------------------------------------------- #

@ad.differentiable
def dot(x, y):

    op = tensor_dot(x, y)

    return op.dot()




# --- Kronecker product ----------------------------------------------------- #

def kron(x, y, kronmap):

    return reindex.fuse(einsum(x, y), kronmap)




