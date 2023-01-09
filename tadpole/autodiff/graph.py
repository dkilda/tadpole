#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

from tadpole.autodiff.node import make_node
from tadpole.autodiff.node import Node, Point

from tadpole.autodiff.util import cacheable
from tadpole.autodiff.util import Stack



###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class Graph:

   _layer = -1


   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def __eq__(self, other):

       return self._fun == other._fun
          and self._x   == other._x


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, root_gate):

       root_node = make_node(self._x, root_gate, type(self)._layer) 
       return self._fun(root_node)




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Gated function -------------------------------------------------------- #

class GatedFun:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __eq__(self, other):

       return self._diff_fun == other._diff_fun
          and self._raw_fun  == other._raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)


   def gate(self, inputs):

       return inputs.transform(self._diff_fun)




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __eq__(self, other):

       return self._fun == other._fun


   def __call__(self, *args):

       glue = make_arg_glue(args, lambda x: isinstance(x, Node))

       return glue.pack()
                  .pluginto(GatedFun(self, self._fun))




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __eq__(self, other):

       return self._fun == other._fun


   def __call__(self, *args):

       glue = make_arg_glue(args, lambda x: isinstance(x, Point))

       return glue.pack()
                  .pluginto(self._fun)




# --- Shorthand decorators for autodiff functions --------------------------- #

def differentiable(fun):
    return Differentiable(fun)


def nondifferentiable(fun):
    return NonDifferentiable(fun)




###############################################################################
###                                                                         ###
###  Argument glue and packs: glueing arguments into a single pack          ###
###                           for function calls.                           ###
###                                                                         ###
###############################################################################


# --- Argument filter ------------------------------------------------------- #

class ArgFilter:

   def __init__(self, mask):

       self._mask = mask


   def __eq__(self, other):

       return self._mask == other._mask


   def _execute(self, mask):

       return tuple(arg for arg in args if mask(arg))


   def vals(self, args):

       return self._execute(lambda x: not self._mask(x))  


   def nodes(self, args):

       return self._execute(lambda x: self._mask(x))  




# --- Glue interface -------------------------------------------------------- #

class Glue(abc.ABC):

   @abc.abstractmethod
   def pack(self, funcall):
       pass




# --- Argument glue --------------------------------------------------------- #

class ArgGlue(Glue):

   def __init__(self, args, argfilter):

       self._args   = args
       self._filter = argfilter


   def __eq__(self, other):

       return self._args   == other._args
          and self._filter == other._filter


   def pack(self, funcall=None):

       if funcall is None:
          funcall = FunCall()

       vals  = self._filter.vals(self._args)
       nodes = self._filter.nodes(self._args)

       funcall = funcall.add(*vals)

       if len(nodes) > 0: 
          return nodes[0].glue(nodes[1:])
                         .pack(funcall)

       return EmptyPack(funcall)




# --- Create argument glue -------------------------------------------------- #

def make_arg_glue(args, mask):

    return ArgGlue(args, ArgFilter(mask))




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, args=None):

       if args is None:
          args = Stack()

       self._args = args


   def __eq__(self, other):

       return self._args == other._args


   def add(self, *args):

       xs = self._args

       for arg in args:
           xs = xs.push(arg)

       return self.__class__(xs)


   def execute(self, fun):

       return fun(*self._args.riter())




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node sources ---------------------------------------------------------- #

class Sources: 

   def __init__(self, nodes, sources, layers):

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __eq__(self, other):

       return self._nodes   == other._nodes
          and self._sources == other._sources
          and self._layers  == other._layers


   @cacheable
   def layer(self):
       return max(self._layers)


   @cacheable
   def adxs(self):
       return tuple(i for i, x in enumerate(self._layers) 
                                      if x == self.layer())

   @cacheable
   def args(self):

       args = list(self._nodes)

       for adx in self._adxs():
           args[adx] = self._sources[adx]

       return args




# --- Node glue ------------------------------------------------------------- #

class NodeGlue(Glue):

   def __init__(self, sources, gates):

       self._sources = sources
       self._gates   = gates


   def __eq__(self, other):

       return self._sources == other._sources
          and self._gates   == other._gates


   def pack(self, funcall):

       source = ArgGlue(self._sources.args()).nodepack(funcall)
       inputs = make_gate_inputs(
                                 self._gates, 
                                 self._sources.adxs(), 
                                 self._sources.args(), 
                                 source
                                )

       return NodePack(source, inputs, self._sources.layer())




# --- Point glue ------------------------------------------------------------ #

class PointGlue(Glue):

   def __init__(self, sources):

       self._sources = sources


   def __eq__(self, other):

       return self._sources == other._sources


   def pack(self, funcall):

       source = ArgGlue(self._sources.args()).pointpack(funcall)

       return PointPack(source, self._sources.layer())




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Pack interface -------------------------------------------------------- #

class Pack(abc.ABC):

   @abc.abstractmethod
   def pluginto(self, fun):
       pass




# --- Node pack ------------------------------------------------------------- #

class NodePack(Pack):

   def __init__(self, source, inputs, layer):

       self._source = source
       self._inputs = inputs
       self._layer  = layer


   def __eq__(self, other):

       return self._sources == other._sources
          and self._inputs  == other._inputs
          and self._layer   == other._layer


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)
       gate   = fun.gate(self._inputs)

       return make_node(source, gate, self._layer)




# --- Point pack ------------------------------------------------------------ #

class PointPack(Pack):

   def __init__(self, source):

       self._source = source


   def __eq__(self, other):

       return self._source == other._source


   def pluginto(self, fun):

       return self._source.pluginto(fun)

      


# --- Empty pack ------------------------------------------------------------ #

class EmptyPack(Pack):

   def __init__(self, funcall):

       self._funcall = funcall


   def __eq__(self, other):

       return self._funcall == other._funcall


   def pluginto(self, fun):

       return self._funcall.execute(fun)































































