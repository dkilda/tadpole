#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import functools

import opt_einsum as oe

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core       as core
import tadpole.tensor.funcall    as fn
import tadpole.tensor.operations as op


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)




###############################################################################
###                                                                         ###
###  Einsum equations                                                       ###
###  (Generating equations from input and output indices,                   ###
###   using unicode or other sets of symbols).                              ###   
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




###############################################################################
###                                                                         ###
###  Index contraction logic:                                               ### 
###  -- generates equation and output indices from input indices            ###
###  -- creates output tensor from data computed by an array product        ###
###     function                                                            ###
###                                                                         ###
###############################################################################


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

      return iter(self._inds)




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




# --- ContractionLike's factory interface ----------------------------------- #

class ContractionLikes(abc.ABC):

   @abc.abstractmethod
   def create(self, inds):
       pass 




# --- Factory of contractions ----------------------------------------------- #

class Contractions(ContractionLikes):

   def __init__(self, product):

       self._product = product


   def create(self, inds):

       return Contraction(inds, self._product)




# --- ContractionLike interface --------------------------------------------- #

class ContractionLike(abc.ABC):

   @abc.abstractmethod
   def equation(self):
       pass 

   @abc.abstractmethod
   def input_inds(self):
       pass 

   @abc.abstractmethod
   def output_inds(self):
       pass 

   @abc.abstractmethod
   def output_tensor(self, data):
       pass 




# --- Contraction ----------------------------------------------------------- #

class Contraction(ContractionLike):

   def __init__(self, inds, product):

       self._inds    = inds
       self._product = product


   def equation(self):

       return make_equation(self._inds, self.output_inds())


   def input_inds(self):

       return iter(self._inds)


   def output_inds(self):

       return self._product(self._inds)


   def output_tensor(self, data):

       return core.Tensor(data, Indices(*self.output_inds()))



"""
# --- Index contraction interface ------------------------------------------- #

class IndexContract(abc.ABC):

   @abc.abstractmethod
   def generate(self, input_inds):
       pass




# --- Fixed index contraction (with fixed output indices) ------------------- #

class FixedIndexContract(IndexContract):

   def __init__(self, output_inds):

       self._output_inds = output_inds


   def generate(self, input_inds):

       return iter(self._output_inds)   




# --- Pair index contraction (finds matching index pairs) ------------------- #

class PairIndexContract(IndexContract):

   def generate(self, input_inds):

       for ind, freq in util.frequencies(input_inds).items():

           if freq > 2:
              raise ValueError(
                 f"{type(self).__name__}: "
                 f"Index {ind} appears more than twice! All input indices: "
                 f"{inds}. If you wish to perform a hyper-index contraction, "
                 f"please specify the output indices explicitly."
              )

           if freq == 1:
              yield ind 

"""


###############################################################################
###                                                                         ###
###  Tensor contraction calls                                               ###
###                                                                         ###
###############################################################################

"""
# --- Conctraction call interface ------------------------------------------- #

class ContractCall(abc.ABC):

   @abc.abstractmethod
   def output_inds(self):
       pass

   @abc.abstractmethod
   def equation(self):
       pass  

"""


# --- Dot product call ------------------------------------------------------ #

class Dot(fn.FunCall):

   def __init__(self, engine, contractions=None):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       if contractions is None:
          contractions = Contractions(PairwiseProduct())

       self._engine       = engine
       self._contractions = contractions


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   def execute(self):

       contract = self._contractions.create(self._engine.inds())  
       data     = self._engine.execute(*self._engine.datas())

       return contract.output_tensor(data)




# --- Einsum call ----------------------------------------------------------- #

class Einsum(fn.FunCall):

   def __init__(self, engine, contractions):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       self._engine       = engine
       self._contractions = contractions


   def attach(self, data, inds):

       return self.__class__(self._engine.attach((data, inds))


   def execute(self):

       contract = self._contractions.create(self._engine.inds()) 
       data     = self._engine.execute(
                                       contract.equation(), 
                                       *self._engine.datas()
                                      )

       return contract.output_tensor(data) 




# --- Specialized contraction methods --------------------------------------- #

@ad.differentiable
def einsum(*xs, product=None, optimize=True):

    if product is None:
       product = PairwiseProduct()

    if not isinstance(product, Product):
       product = FixedProduct(product)

    def fun(equation, *datas):
        return ar.einsum(equation, *datas, optimize=optimize)

    return funcall.execute(Einsum(fun, Contractions(product)))




@ad.differentiable
def dot(x, y):

    def fun(u, v):
        return ar.dot(u, v)

    return funcall.execute(Dot(fun))




def kron(x, y, kronmap):

    out = einsum(x, y)

    return op.fuse(out, kronmap)




