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
###  Index contraction                                                      ###
###  (different strategies to generate output indices from input indices)   ###
###                                                                         ###
###############################################################################


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




###############################################################################
###                                                                         ###
###  Contraction calls                                                      ###
###                                                                         ###
###############################################################################


# --- Conctraction call interface ------------------------------------------- #

class ContractCall(abc.ABC):

   @abc.abstractmethod
   def output_inds(self):
       pass

   @abc.abstractmethod
   def equation(self):
       pass  




# --- Dot product call ------------------------------------------------------ #

class Dot(fn.FunCall, ContractCall):

   def __init__(self, engine, indcon=None):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       if indcon is None:
          indcon = PairIndexContract()

       self._engine = engine
       self._indcon = indcon


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   @util.cacheable
   def output_inds(self):

       return self._indcon.generate(self._engine.inds()) 


   @util.cacheable
   def equation(self):

       return make_equation(self._engine.inds(), self.output_inds())  


   def execute(self):

       outdata = self._engine.execute(*self._engine.datas())

       return core.Tensor(outdata, self.output_inds())




# --- Einsum call ----------------------------------------------------------- #

class Einsum(fn.FunCall, ContractCall):

   def __init__(self, engine, indcon):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       self._engine = engine
       self._indcon = indcon


   def attach(self, data, inds):

       return self.__class__(self._engine.attach((data, inds))


   @util.cacheable
   def output_inds(self):
 
       return self._indcon.generate(self._engine.inds())  


   @util.cacheable
   def equation(self):

       return make_equation(self._engine.inds(), self.output_inds())  


   def execute(self):

       outdata = self._engine.execute(self.equation(), *self._engine.datas())

       return core.Tensor(outdata, self.output_inds())  




# --- Linear algebra: contraction methods ----------------------------------- #

@ad.differentiable
def einsum(*xs, outinds=None, optimize=True):

    if outinds is None:
       outinds = PairIndexContract()

    if not isinstance(outinds, IndexContract):
       outinds = FixedIndexContract(outinds)

    def fun(equation, *datas):
        return ar.einsum(equation, *datas, optimize=optimize)

    return fn.Args(*xs).pluginto(Einsum(fun, outinds))




@ad.differentiable
def dot(x, y):

    def fun(u, v):
        return ar.dot(u, v)

    return fn.Args(x, y).pluginto(Dot(fun))




def kron(x, y, kronmap):

    out = einsum(x, y)

    return op.fuse(out, kronmap)




