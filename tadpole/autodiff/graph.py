#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.util as util
from tadpole.util import TupleLike

import tadpole.autodiff.node as anode




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

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("root", self._root)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.ref(self._root, other._root)

       return bool(log)
       

   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, fun, x):

       start = anode.Node(x, type(self)._layer, self._root) 

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

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args, **kwargs):

       envelope = self._envelope(*args, **kwargs)
       out      = envelope.applywrap(self, self._fun)

       return out.unpack() 




# --- Non-differentiable function wrap -------------------------------------- #

class NonDifferentiable:

   def __init__(self, fun, envelope):

       self._fun      = fun
       self._envelope = envelope


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args, **kwargs):

       out = self._envelope(*args, **kwargs).apply(self._fun)

       return out.unpack()



# --- Shorthand for a differentiable function wrap -------------------------- #

def differentiable(fun):

    def envelope(*args, **kwargs):
        return Envelope(*args, **kwargs)

    return Differentiable(fun, envelope)




# --- Shorthand for a non-differentiable function wrap ---------------------- #

def nondifferentiable(fun):

    def envelope(*args, **kwargs):
        return Envelope(*args, **kwargs)

    return NonDifferentiable(fun, envelope)




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

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("args", self._args)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.val(self._args, other._args)

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

           if isinstance(arg, anode.NodeLike):
              yield arg
              continue

           yield anode.Point(arg)


   @util.cacheable
   def concat(self):

       concat = Concatenation() 

       for node in self.nodify():
           concat = node.concat(concat)

       return concat

 
   @util.cacheable
   def pack(self):

       return Pack(self.concat())


   def deshelled(self):

       args = self.concat().deshell()

       if args.concat().innermost():
          return args.concat().deshell()

       return args




# --- Concatenable interface ------------------------------------------------ #

class Concatenable(abc.ABC):

   @abc.abstractmethod
   def attach(self, node, source, layer):
       pass




# --- Cohesive interface ---------------------------------------------------- #

class Cohesive(abc.ABC):

   @abc.abstractmethod
   def innermost(self):
       pass

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

       if nodes   is None: nodes   = util.Sequence()
       if sources is None: sources = util.Sequence()
       if layers  is None: layers  = util.Sequence()

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers  


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("nodes",   self._nodes)
       rep.val("sources", self._sources)
       rep.val("layers",  self._layers)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

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


   @util.cacheable
   def innermost(self):

       return self.layer() == minlayer() 


   @util.cacheable
   def layer(self):

       return max(self._layers)


   @util.cacheable
   def adxs(self):

       if self.innermost():
          return tuple()

       return tuple( 
          i for i,x in enumerate(self._layers) if x == self.layer()
       )


   @util.cacheable
   def parents(self):

       nodes = list(self._nodes)
       nodes = [nodes[adx] for adx in self.adxs()] 

       return anode.Parents(*nodes)


   @util.cacheable
   def deshell(self):

       if self.innermost():
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

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("concat", self._concat)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

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

       return self._concat.innermost() 


   def deshell(self):

       return self._concat.deshell()


   def deshelled(self):

       return self._args.pack()


   def _fold(self, funwrap, outputs, out):

       if self.innermost():
          return anode.Point(out)

       op = anode.AdjointOp(funwrap, self._adxs, outputs, self._args)

       return self._parents.next(out, self._layer, op) 


   def fold(self, funwrap, outputs):

       def _fold(out):
           return self._fold(funwrap, outputs, out)

       return outputs.apply(_fold)


       

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

   def __init__(self, *args, **kwargs):

       if len(args) == 0:
          raise ValueError("Envelope must have at least one input")

       if len(args) == 1 and isinstance(args[0], ArgsLike):
          args, = args
          
       if not isinstance(args, ArgsLike):
          args = Args(*args)
            
       self._args   = args
       self._kwargs = kwargs


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("args",   self._args)
       rep.ref("kwargs", self._kwargs)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other) 
       log.ref(self._args,   other._args)
       log.ref(self._kwargs, other._kwargs)

       return bool(log)


   def __hash__(self):

       return id(self)

 
   def packs(self):

       return util.Loop(
                        self._args.pack(),  
                        lambda x: x.deshelled(), 
                        lambda x: x.innermost()
                       )


   def apply(self, fun):

       last = self.packs().last()
       args = last.deshell() 

       return fun(*args, **self._kwargs) # TODO solution: require any ops function to return a tuple/TupleLike/Outputs?

              
   def applywrap(self, funwrap, fun): 

       if self.packs().once():
          return self.apply(fun)
           
       out = self.apply(fun)

       for pack in reversed(self.packs()): 
           out = pack.fold(funwrap, out)

       return out




