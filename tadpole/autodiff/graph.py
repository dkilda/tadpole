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
   Envelope
)




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


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

       return fun(start)




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
       out      = envelope.applywrap(self, self._fun)

       return out.unpack() 


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

       out = self._envelope(*args, **kwargs).apply(self._fun)

       return out.unpack()




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

       return len(self._args)


   def __contains__(self, x):

       return x in self._args


   def __iter__(self):

       return iter(self._args)


   def __getitem__(self, idx):

       return self._args[idx]


   def nodify(self):

       for arg in self._args:

           if isinstance(arg, an.Node):
              yield arg
              continue

           yield an.point(arg)


   @util.cacheable
   def concat(self):

       concat = ConcatArgs() 

       for node in self.nodify():
           concat = node.concat(concat)

       return concat

 
   def pack(self, **kwargs):

       return PackArgs(self.concat(), **kwargs)


   def deshelled(self):

       args = self.concat().deshell()

       if args.concat().innermost():
          return args.concat().deshell()

       return args




# --- Concatenation of arguments -------------------------------------------- #

class ConcatArgs(Sequential, Cohesive):

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
          i for i,x in enumerate(self._layers) if x == self.layer()
       )


   @util.cacheable
   def parents(self):

       nodes = list(self._nodes)
       nodes = [nodes[adx] for adx in self.adxs()] 

       return an.ParentsGen(*nodes)


   @util.cacheable
   def deshell(self):

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


   def _fold(self, funwrap, outputs, out):

       if self.innermost():
          return an.point(out)

       op = an.AdjointOpGen(
          funwrap, self._adxs, outputs, self._args, self._kwargs
       )

       return self._parents.next(out, self._layer, op) 


   def fold(self, funwrap, outputs):

       def _fold(out):
           return self._fold(funwrap, outputs, out)

       return outputs.apply(_fold)




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




