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





@functools.lru_cache(2**12)
def unicode_symbol(ind):

    return oe.get_symbol(ind)




class Symbols:
 
   def __init__(self, symbols):

       self._symbols = symbols


   @util.cacheable
   def _map(self):

       next_symbol = map(self._symbols, itertools.count()).__next__

       return collections.defaultdict(next_symbol)


   def next(self, ind):

       return self._map()[ind]




class Equation:

   def __init__(self, symbols=None):

       if symbols is None:
          symbols = Symbols(unicode_symbol)

       self._symbols = symbols

   
   @functools.lru_cache(2**12)
   def tosymbols(self, inds):

       return map(self._symbols.next, inds)


   @functools.lru_cache(2**12)
   def make(self, input_inds, output_inds): 

       inputs = tuple("".join(self.tosymbols(inds)) for inds in input_inds)
       output =       "".join(self.tosymbols(output_inds)) 

       return ",".join(inputs) + f"->{output}"

       




class IndexContract(abc.ABC):

   @abc.abstractmethod
   def equation(self):
       pass

   @abc.abstractmethod
   def input(self):
       pass

   @abc.abstractmethod
   def output(self):
       pass



class FixedIndexContract(IndexContract):

   def __init__(self, input_inds, output_inds, equation=None):

       if equation is None:
          equation = Equation()

       self._input    = input_inds
       self._output   = output_inds
       self._equation = equation


   def equation(self):

       return self._equation.make(self._input, self._output)


   def input(self):

       return iter(self._input)


   def output(self):

       return iter(self._output)





class DotIndexContract(IndexContract):

   def __init__(self, inds, equation=None):

       if equation is None:
          equation = Equation()

       self._inds     = inds
       self._equation = equation


   def _frequencies(self):

       return iter(util.frequencies(inds).items())


   def equation(self):

       return self._equation.make(self._inds, self.output())


   def input(self):

       return iter(self._inds)


   def output(self):

       output = []

       for ind, freq in self._frequencies(): 

           if freq > 2:
              raise ValueError(
                 f"{type(self).__name__}.output: "
                 f"Index {ind} appears more than twice! All input indices: "
                 f"{inds}. If you wish to perform a hyper-index contraction, "
                 f"please specify the output indices explicitly."
              )

           if freq == 1:
              yield ind 







"""
@functools.lru_cache(2**12)
def make_equation(input_inds, output_inds):

    symbols = Symbols(unicode_symbol)

    inputs = tuple("".join(map(symbols.next, inds)) for inds in input_inds)
    output =       "".join(map(symbols.next, output_inds))

    return ",".join(inputs) + f"->{output}"

    


def make_output_inds(inds):

    output = []

    for ind, freq in util.frequencies(inds).items():

        if freq > 2:
           raise ValueError(
              f"make_output_inds: "
              f"Index {ind} appears more than twice! All "
              f"input indices: {inds}. If you wish to perform a hyper-index "
              f"contraction, please specify the output indices explicitly."
           )

        if freq == 1:
           output.append(ind)

    return tuple(output)

"""








###############################################################################
###                                                                         ###
###  Conctraction calls                                                     ###
###                                                                         ###
###############################################################################


# --- Dot product call ------------------------------------------------------ #

class Dot(fn.FunCall):

   def __init__(self, engine):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       self._engine = engine


   def attach(self, data, inds):

       return self.__class__(self._engine.attach(data, inds))


   @util.cacheable
   def outinds(self):

       return make_output_inds(self._engine.inds())


   def execute(self):

       outdata = self._engine.execute(*self._engine.datas())

       return core.Tensor(outdata, self.outinds())




# --- Einsum call ----------------------------------------------------------- #

class Einsum(fn.FunCall):

   def __init__(self, engine, indcon):

       if not isinstance(engine, fn.EngineLike):
          engine = fn.Engine(engine)

       self._engine = engine
       self._indcon = indcon


   def attach(self, data, inds):

       return self.__class__(self._engine.attach((data, inds))


   @util.cacheable
   def outinds(self):
 
       self._indcon.output()

       if self._outinds is None: 
          return make_output_inds(self._engine.inds())

       return self._outinds


   @util.cacheable
   def equation(self):

       return make_equation(self._engine.inds(), self.outinds())


   def execute(self):

       outdata = self._engine.execute(
                    self.equation(), *self._engine.datas()
                 )

       return core.Tensor(outdata, self.outinds())  










# --- Linear algebra: multiplication methods -------------------------------- #

@ad.differentiable
def einsum(*xs, outinds=None, optimize=True):

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























































