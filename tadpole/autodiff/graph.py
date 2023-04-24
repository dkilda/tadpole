#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util          as util
import tadpole.autodiff.misc as misc
import tadpole.autodiff.node as an


from tadpole.autodiff.types import (
   DifferentiableFun, 
   Args,
   Sequential,
   Cohesive,
   Pack,
   Envelope,
   Node,
)




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Helper: cast function return type to Node ----------------------------- #

def returns_node(fun):

    def wrap(*args, **kwargs):

        out = fun(*args, **kwargs) 

        if not isinstance(out, Node): 
           return an.point(out)

        return out

    return wrap





# --- Graph ----------------------------------------------------------------- #

class Graph:

   _layer = misc.minlayer() 


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

       if bool(log):
          log.ref(self._root, other._root)

       return bool(log)
       

   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, fun, x):

       start = an.node(x, type(self)._layer, self._root)  
       end   = returns_node(fun)(start)  

       return start, end




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable function wrap ------------------------------------------ #

class Differentiable(DifferentiableFun):

   def __init__(self, fun, envelope, vjpmap, jvpmap):

       self._fun      = fun
       self._envelope = envelope
       self._vjpmap   = vjpmap
       self._jvpmap   = jvpmap


   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       rep.val("fun", self._fun)

       return str(rep)


   def __call__(self, *args, **kwargs):

       envelope = self._envelope(*args, **kwargs)

       return envelope.applywrap(self, self._fun)


   def vjp(self, *args, **kwargs):

       return self._vjpmap.get(self)(*args, **kwargs)


   def jvp(self, *args, **kwargs):

       return self._jvpmap.get(self)(*args, **kwargs)




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

       envelope = self._envelope(*args, **kwargs)

       return envelope.apply(self._fun)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- General function arguments -------------------------------------------- #

class ArgsGen(Args):  

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

       if bool(log):
          log.val(self._args, other._args)

       return bool(log)


   def __hash__(self):

       return hash(self._args)


   def __len__(self):

       return len(self._denodified())


   def __contains__(self, x):

       return x in self._denodified()


   def __iter__(self):

       return iter(self._denodified())


   def __getitem__(self, idx):

       return self._denodified()[idx]


   @util.cacheable
   def _nodified(self):

       return nodify(*self._args)


   @util.cacheable
   def _denodified(self):

       return denodify(*self._nodified())


   def concat(self):

       return ConcatArgs(train_args(*self._nodified())) 

 
   def pack(self, **kwargs):

       return PackArgs(self.concat(), **kwargs)


   def deshelled(self):

       args = self.concat().deshell()

       if args.concat().innermost():
          return args.concat().deshell()

       return args




# --- Helper: convert any plain values to bottom-layer Nodes ---------------- #

def nodify(*args):

    def _nodify(*args):

       for arg in args:

           if isinstance(arg, an.Node):
              yield arg
              continue

           yield an.point(arg)

    return tuple(_nodify(*args))




# --- Helper: convert the bottom-layer Nodes to plain values ---------------- #

def denodify(*nodes):

    train   = train_args(*nodes)
    args    = list(train.nodes())
    sources = list(train.sources())

    for i, layer in enumerate(train.layers()):
        if layer == misc.minlayer():
           args[i] = sources[i]
   
    return tuple(args)




# --- Create a train of arguments ------------------------------------------- #

def train_args(*nodes):

    train = TrainArgs()
    for node in nodes:
        train = node.concat(train)

    return train




# --- Train of arguments ---------------------------------------------------- #
 
class TrainArgs(Sequential):

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

       if bool(log):
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

   def nodes(self):

       return iter(self._nodes)


   def sources(self):

       return iter(self._sources)


   def layers(self):

       return iter(self._layers)




# --- Concatenation of arguments -------------------------------------------- #

class ConcatArgs(Sequential, Cohesive):

   def __init__(self, *args, train=None, **kwargs):

       if train is None and len(args) > 0:
          train = args[0]

       if not isinstance(train, Sequential):
          train = TrainArgs(*args, **kwargs) 

       self._train = train


   def __repr__(self):

       rep = util.ReprChain()
       rep.typ(self)
       rep.val("train", self._train)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
          log.val(self._train, other._train)

       return bool(log)


   def __hash__(self):

       return hash(self._train)


   def attach(self, node, source, layer):

       return self.__class__(self._train.attach(node, source, layer))


   @property
   def _nodes(self):

       return iter(self._train.nodes())


   @property
   def _sources(self):

       return iter(self._train.sources())


   @property
   def _layers(self):

       return iter(self._train.layers())


   @util.cacheable
   def innermost(self):

       return self.layer() == misc.minlayer() 


   @util.cacheable
   def layer(self):

       return max(self._layers)


   @util.cacheable
   def adxs(self):

       if self.innermost():
          return tuple()

       return tuple( 
          i for i, x in enumerate(self._layers) if x == self.layer()
       )


   @util.cacheable
   def parents(self):

       nodes = list(self._nodes)
       nodes = [nodes[adx] for adx in self.adxs()] 

       return an.ParentsGen(*nodes)


   #@util.cacheable
   def deshell(self):

       print("DESHELL: ", tuple(self._layers), self.layer(), self.adxs())

       if self.innermost():
          return ArgsGen(*self._sources)

       args    = list(self._nodes)
       sources = list(self._sources)

       for adx in self.adxs():
           args[adx] = sources[adx] 

       return ArgsGen(*args)

 


###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack (of concatenated nodes) --------------------------------- #

class PackArgs(Pack):

   def __init__(self, concat, **kwargs):

       self._concat = concat
       self._kwargs = kwargs


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.ref("concat", self._concat)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other) 

       if bool(log):
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

       return self._args.pack(**self._kwargs)


   def fold(self, funwrap, out):

       if self.innermost():
          return an.point(out)

       try:
          print("FOLD: ", funwrap._fun.__name__, self._adxs, self._args)
       except AttributeError:
          pass

       op = an.AdjointOpGen(
          funwrap, self._adxs, out, self._args, self._kwargs
       )

       return self._parents.next(out, self._layer, op)




# --- Argument envelope ----------------------------------------------------- #

class EnvelopeArgs(Envelope):

   def __init__(self, *args, **kwargs):

       if len(args) == 0:
          raise ValueError(
             f"{type(self).__name__} must have at least one input"
          )

       if len(args) == 1 and isinstance(args[0], Args):
          args, = args
          
       if not isinstance(args, Args):
          args = ArgsGen(*args)
            
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

       if bool(log):
          log.ref(self._args,   other._args)
          log.ref(self._kwargs, other._kwargs)

       return bool(log)


   def __hash__(self):

       return id(self)

 
   def packs(self):

       return util.Loop(
                        self._args.pack(**self._kwargs),  
                        lambda x: x.deshelled(), 
                        lambda x: x.innermost()
                       )


   def apply(self, fun):

       last = self.packs().last()
       args = last.deshell() 

       return fun(*args, **self._kwargs) 

              
   def applywrap(self, funwrap, fun): 

       if self.packs().once():
          return self.apply(fun)
           
       out = self.apply(fun)

       for pack in reversed(self.packs()): 
           out = pack.fold(funwrap, out)

       return out




