#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.autodiff.node as tdn

from tadpole.autodiff.util import cacheable, Stack, StringRep




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


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, root_gate):

       root_node = tdn.make_node(self._x, root_gate, type(self)._layer) 
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


   def __call__(self, *args):

       return self._raw_fun(*args)


   def gate(self, inputs):

       return inputs.transform(self._diff_fun)




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       glue = ArgGlue(args, ArgFilterByNode())

       return glue.pack().pluginto(GatedFun(self, self._fun))




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       glue = ArgGlue(args, ArgFilterByPoint())

       return glue.pack().pluginto(self._fun)




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


   def _str(self):

       out = StringRep(self)
       out = out.with_member("mask", self._mask)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._mask == other._mask


   def _execute(self, args, mask):

       return tuple(arg for arg in args if mask(arg))


   def vals(self, args):

       return self._execute(args, lambda x: not self._mask(x))  


   def nodes(self, args):

       return self._execute(args, lambda x: self._mask(x))  




class ArgFilterByNode(ArgFilter):

   def __init__(self):

       super().__init__(lambda x: isinstance(x, tdn.Node))




class ArgFilterByPoint(ArgFilter):

   def __init__(self):

       super().__init__(lambda x: isinstance(x, tdn.Point))




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


   def _str(self):

       out = StringRep(self)
       out = out.with_member("args",   self._args)
       out = out.with_member("filter", self._filter)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._args   == other._args \
          and self._filter == other._filter


   def pack(self, funcall=None):

       if funcall is None:
          funcall = FunCall()

       vals  = self._filter.vals(self._args)
       nodes = self._filter.nodes(self._args)

       funcall = funcall.add(*vals)

       if len(nodes) > 0: 
          return nodes[0].glue(*nodes[1:]).pack(funcall)

       return EmptyPack(funcall)




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, args=None):

       if args is None:
          args = Stack()

       self._args = args


   def _str(self):

       out = StringRep(self)
       out = out.with_member("args", self._args)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


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


   def _str(self):

       out = StringRep(self)
       out = out.with_data("layers",    self._layers)
       out = out.with_member("nodes",   self._nodes)
       out = out.with_member("sources", self._sources)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._nodes   == other._nodes   \
          and self._sources == other._sources \
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

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return args




# --- Node glue ------------------------------------------------------------- #

class NodeGlue(Glue):

   def __init__(self, sources, gates):

       self._sources = sources
       self._gates   = gates


   def _str(self):

       out = StringRep(self)
       out = out.with_member("sources", self._sources)
       out = out.with_member("gates",   self._gates)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._sources == other._sources \
          and self._gates   == other._gates


   def pack(self, funcall):

       glue   = ArgGlue(self._sources.args(), ArgFilterByNode())
       source = glue.pack(funcall)
       inputs = tdn.make_gate_inputs(
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


   def _str(self):

       out = StringRep(self)
       out = out.with_member("sources", self._sources)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._sources == other._sources


   def pack(self, funcall):

       glue   = ArgGlue(self._sources.args(), ArgFilterByPoint())
       source = glue.pack(funcall)  

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


   def _str(self):

       out = StringRep(self)
       out = out.with_data("layer",    self._layer)
       out = out.with_member("source", self._source)
       out = out.with_member("inputs", self._inputs)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._sources == other._sources \
          and self._inputs  == other._inputs  \
          and self._layer   == other._layer


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)
       gate   = fun.gate(self._inputs)

       return tdn.make_node(source, gate, self._layer)




# --- Point pack ------------------------------------------------------------ #

class PointPack(Pack):

   def __init__(self, source):

       self._source = source


   def _str(self):

       out = StringRep(self)
       out = out.with_member("source", self._source)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._source == other._source


   def pluginto(self, fun):

       return self._source.pluginto(fun)

      


# --- Empty pack ------------------------------------------------------------ #

class EmptyPack(Pack):

   def __init__(self, funcall):

       self._funcall = funcall


   def _str(self):

       out = StringRep(self)
       out = out.with_member("funcall", self._funcall)

       return out.compile()
       

   def __str__(self):
 
       return self._str()


   def __repr__(self):

       return self._str()


   def __eq__(self, other):

       return self._funcall == other._funcall


   def pluginto(self, fun):

       return self._funcall.execute(fun)































































