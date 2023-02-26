#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.autodiff.util as tdutil
import tadpole.autodiff.node as tdnode

from tadpole.autodiff.util import TupleLike




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Define the minimum node layer ----------------------------------------- #

def minlayer():

    return -1




# --- GraphLike interface --------------------------------------------------- #

class GraphLike(abc.ABC):

   @abc.abstractmethod
   def __enter__(self):
       pass

   @abc.abstractmethod
   def __exit__(self, exception_type, exception_val, trace):
       pass

   @abc.abstractmethod
   def build(self, fun, x):
       pass




# --- Graph ----------------------------------------------------------------- #

class Graph(GraphLike):

   _layer = minlayer() 


   def __init__(self, root):

       self._root = root


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("root", self._root)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other)
       log.ref(self._root, other._root)

       return bool(log)
       

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

   def __init__(self, fun, envelope):

       self._fun      = fun
       self._envelope = envelope


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args):

       return self._envelope(*args).applywrap(self, self._fun)




# --- Non-differentiable function wrap -------------------------------------- #

class NonDifferentiable:

   def __init__(self, fun, envelope):

       self._fun      = fun
       self._envelope = envelope


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args):

       return self._envelope(*args).apply(self._fun)




# --- Shorthand for a differentiable function wrap -------------------------- #

def differentiable(fun):

    return Differentiable(fun, lambda *args: Envelope(*args))




# --- Shorthand for a non-differentiable function wrap ---------------------- #

def nondifferentiable(fun):

    return NonDifferentiable(fun, lambda *args: Envelope(*args))




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- ArgsLike interface ---------------------------------------------------- #

class ArgsLike(abc.ABC):

   @abc.abstractmethod
   def concat(self):
       pass

   @abc.abstractmethod
   def pack(self):
       pass




# --- Function arguments ---------------------------------------------------- #

class Args(ArgsLike, TupleLike):  

   def __init__(self, *args):

       self._args = args


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.ref(self._args, other._args)

       return bool(log)


   def __hash__(self):

       return hash(self._args)


   def __len__(self):

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __getitem__(self, idx):

       return self._args[idx]


   def nodify(self):

       for arg in self._args:

           if isinstance(arg, tdnode.NodeLike):
              yield arg
              continue

           yield tdnode.Point(arg)


   def concat(self):

       concat = Concatenation() 

       for node in self.nodify():
           concat = node.concat(concat)

       return concat

 
   def pack(self):

       return Pack(self.concat())


   def deshelled(self):

       return self.concat().deshell()




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

       log = tdutil.LogicalChain()

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

       if self.layer() == minlayer():
          return Args(*self._sources)

       args    = list(self._nodes)
       sources = list(self._sources)

       for adx in self.adxs():
           args[adx] = sources[adx] 

       return Args(*args)




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Packable interface ---------------------------------------------------- #

class Packable(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

   @abc.abstractmethod
   def deshell(self):
       pass

   @abc.abstractmethod
   def deshelled(self):
       pass

   @abc.abstractmethod
   def fold(self, funwrap, out):
       pass




# --- Argument pack (of concatenated nodes) --------------------------------- #

class Pack(Packable):

   def __init__(self, concat):

       self._concat = concat


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("concat", self._concat)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other) 
       log.val(self._concat, other._concat)

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


   def deshell(self):

       return self._concat.deshell()


   def deshelled(self):

       return self._args.pack()

       
   def fold(self, funwrap, out): 

       if self.innermost(): 
          return tdnode.Point(out)

       op = tdnode.AdjointOp(funwrap, self._adxs, out, self._args)
       return self._parents.next(out, self._layer, op) 




# --- EnvelopeLike interface ------------------------------------------------ #

class EnvelopeLike(abc.ABC): 

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

class Envelope(EnvelopeLike):

   def __init__(self, *args):

       if len(args) == 0:
          raise ValueError("Envelope must have at least one input")

       if not isinstance(args[0], ArgsLike):
          args = Args(*args)

       self._args = args


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

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

       try:
            last = self.packs().last()
       except StopIteration:
            last = self.packs().first()

       args = last.deshell() 
       out  = fun(*args)

       return out      

       
   def applywrap(self, funwrap, fun):

       out = self.apply(fun)

       for pack in reversed(self.packs()):
           out = pack.fold(funwrap, out) 

       if isinstance(out, tdnode.Point):
          return out.tovalue() # FIXME Interim solution: in general, applywrap() must always return
                               # the same type (NodeLike), so this will get fixed once our Value/Array = Point.
       return out




