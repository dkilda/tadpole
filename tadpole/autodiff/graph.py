#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.adjoint as tda
import tadpole.autodiff.util    as tdutil
import tadpole.autodiff.node    as tdnode




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Define the minimum node layer ----------------------------------------- #

def minlayer():

    return -1




# --- Graph ----------------------------------------------------------------- #

class Graph:

   _layer = minlayer() 


   def __init__(self, root):

       self._root = root


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, fun, x):

       start = tdnode.Node(x, type(self)._layer, self._root) 

       return fun(start)




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable function wrap ------------------------------------------ #

class Differentiable:

   def __init__(self, fun, make_envelope):

       self._fun      = fun
       self._envelope = make_envelope


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args):

       return self._envelope(args).applywrap(self, self._fun)




# --- Non-differentiable function wrap -------------------------------------- #

class NonDifferentiable:

   def __init__(self, fun, make_envelope):

       self._fun      = fun
       self._envelope = make_envelope


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args):

       return self._envelope(args).apply(self._fun)




# --- Shorthand for a differentiable function wrap -------------------------- #

def differentiable(fun):

    return Differentiable(fun, lambda x: Envelope(x))




# --- Shorthand for a non-differentiable function wrap ---------------------- #

def nondifferentiable(fun):

    return NonDifferentiable(fun, lambda x: Envelope(x))




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- Compound interface ---------------------------------------------------- #

class Compound(abc.ABC):

   @abc.abstractmethod
   def concatenate(self):
       pass

   @abc.abstractmethod
   def pack(self):
       pass




# --- Helpers for Args ------------------------------------------------------- #

def nodify(x):

    if isinstance(x, tdnode.NodeLike):
       return x

    return tdnode.Point(x)




# --- Function arguments ---------------------------------------------------- #

class Args(tdutil.Tuple):


   def concat(self):

       concat = Concatenation() 
       args   = map(nodify, self._xs)

       for arg in args:
           concat = arg.concat(concat)

       return concat

 
   def pack(self):

       return Pack(self.concat())




# --- Concatenable interface ------------------------------------------------ #

class Concatenable(abc.ABC):

   @abc.abstractmethod
   def attach(self, node, source, layer):
       pass




# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(abc.ABC):

   @abc.abstractmethod
   def layer(self):
       pass

   @abc.abstractmethod
   def adxs(self):
       pass

   @abc.abstractmethod
   def parents(self):
       pass

   @abc.abstractmethod
   def deshell(self):
       pass




# --- Concatenation of nodes ------------------------------------------------ #

class Concatenation(Concatenable, Cohesive):

   def __init__(self, nodes=None, sources=None, layers=None):

       if nodes   is None: nodes   = tdutil.Sequence()
       if sources is None: sources = tdutil.Sequence()
       if layers  is None: layers  = tdutil.Sequence()

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("nodes",   self._nodes)
       rep.val("sources", self._sources)
       rep.val("layers",  self._layers)

       return str(rep)


   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.val(self._nodes,   other._nodes)
       log.val(self._sources, other._sources)
       log.val(self._layers,  other._layers)

       return bool(log)


   def __hash__(self):

       return id(self)


   def attach(self, node, source, layer):

       return self.__class__(
                             self._nodes.push(node), 
                             self._sources.push(source), 
                             self._layers.push(layer)
                            )

   @tdutil.cacheable
   def layer(self):

       return max(self._layers)


   @tdutil.cacheable
   def adxs(self):

       if self.layer() == minlayer():
          return tuple()

       return tuple(i for i, x in enumerate(self._layers) 
                                           if x == self.layer())

   @tdutil.cacheable
   def parents(self):

       nodes = list(self._nodes)
       nodes = [nodes[adx] for adx in self.adxs()] 
       return tdnode.Parents(nodes)


   @tdutil.cacheable
   def deshell(self):

       args    = list(self._nodes)
       sources = list(self._sources)

       for adx in self.adxs():
           args[adx] = sources[adx] 

       return Args(args)




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Packed interface ------------------------------------------------------ #

class Packed(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

   @abc.abstractmethod
   def deshelled(self):
       pass

   @abc.abstractmethod
   def fold(self, funwrap, out):
       pass




# --- Argument pack (of concatenated nodes) --------------------------------- #

class Pack(Packed):

   def __init__(self, concat): 

       self._concat = concat


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("concat", self._concat)

       return str(rep)


   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.ref(self._concat, other._concat)

       return bool(log)


   def __hash__(self):

       return id(self)


   @property
   def _layer(self):

       return self._concat.layer()


   @property
   def _adxs(self):

       return self._concat.adxs()


   @property
   def _args(self):

       return self._concat.deshell()


   @property
   def _parents(self):

       return self._concat.parents()


   def innermost(self):

       return self._layer == minlayer()


   def deshelled(self):

       return self.__class__(self._args.pack())

       
   def fold(self, funwrap, out): 

       if self.innermost(): 
          return tdnode.Point(out)

       op = tdnode.AdjointOp(funwrap, self._adxs, out, self._args)
       return self._parents.next(out, self._layer, op) 




# --- Enveloped interface --------------------------------------------------- #

class Enveloped(abc.ABC):

   @abc.abstractmethod
   def packs(self):
       pass

   @abc.abstractmethod
   def apply(self, fun):
       pass

   @abc.abstractmethod
   def applywrap(self, funwrap, out):
       pass




# --- Argument envelope ----------------------------------------------------- #

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.


class Envelope(Enveloped):

   def __init__(self, args):

       self._args = args


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = LogicalChain()

       log.typ(self, other) 
       log.ref(self._args, other._args)

       return bool(log)


   def __hash__(self):

       return id(self)

 
   def packs(self):

       return tdutil.Loop(
                          self._args.pack(),  
                          lambda x: x.deshelled(), 
                          lambda x: x.innermost()
                         )

   def apply(self, fun):

       args = self.packs().last().args()
       out  = fun(*(arg.tovalue() for arg in args))

       return tdnode.Point(out)     

       
   def applywrap(self, funwrap, fun):

       out = self.apply(fun)

       for pack in reversed(self.packs()):
           out = pack.fold(funwrap, out) 

       return out




