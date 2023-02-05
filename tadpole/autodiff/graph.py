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


   def __init__(self, fun, x):

       self._fun = fun
       self._x   = x


   def __enter__(self):

       type(self)._layer += 1
       return self


   def __exit__(self, exception_type, exception_val, trace):

       type(self)._layer -= 1


   def build(self, gate):

       root = tdnode.make_node(self._x, type(self)._layer, gate) 

       return self._fun(root)




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Adjointable Function -------------------------------------------------- #

class AdjointableFun:

   def __init__(self, diff_fun, raw_fun):

       self._diff_fun = diff_fun
       self._raw_fun  = raw_fun


   def __call__(self, *args):

       return self._raw_fun(*args)

  
   def vjp(self, *args, **kwargs): 

       return tda.vjpmap.get(self._diff_fun)(*args, **kwargs)


   def jvp(self, *args, **kwargs): 

       return tda.jvpmap.get(self._diff_fun)(*args, **kwargs)




# --- Differentiable function decorator ------------------------------------- #

class Differentiable: 

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       fun         = AdjointableFun(self, self._fun)
       concat_args = Active(ConcatArgs(args)) 

       return (
               concat_args.pack()
                          .pluginto(fun) 
              )




# --- Non-differentiable function decorator --------------------------------- #

class NonDifferentiable:

   def __init__(self, fun):

       self._fun = fun


   def __call__(self, *args):

       concat_args = Passive(ConcatArgs(args))

       return (
               concat_args.pack()
                          .pluginto(self._fun)
              )




# --- Shorthand decorators for autodiff functions --------------------------- #

def differentiable(fun):

    return Differentiable(fun)




def nondifferentiable(fun):

    return NonDifferentiable(fun)




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

class NodeTrain:

   def __init__(self, nodes=None, meta=None):

       if nodes is None: nodes = tdutil.Sequence()       
       if meta  is None: meta  = tdutil.Sequence()

       self._nodes = nodes
       self._meta  = meta


   def __eq__(self, other):

       return all((
                   self._nodes == other._nodes,  
                   self._meta  == other._meta,
                 ))


   def __hash__(self):

       return id(self)


   def with_node(self, node):

       return self.__class__(self._nodes.push(node), self._meta)


   def with_meta(self, source, layer): 
 
       return self.__class__(self._nodes, self._meta.push((source, layer)))

       
   def concatenate(self):

       try:
          sources, layers = zip(*self._meta)
       except ValueError:
          sources, layers = tuple(), tuple()

       return ConcatArgsKernel(tuple(self._nodes), sources, layers)




# --- Node glue ------------------------------------------------------------- #

class NodeGlue:

   def __init__(self, args):

       self._args = args


   def iterate(self):

       return iter(map(tdnode.nodify, self._args))


   def concatenate(self):

       train = NodeTrain()

       for arg in self.iterate():
           train = arg.attach(train)

       return train.concatenate()




###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


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


   @abc.abstractmethod
   def deshelled(self):
       pass




# --- Packable interface ---------------------------------------------------- #

class Packable(abc.ABC):

   @abc.abstractmethod
   def pack(self):
       pass




# --- Concatenated arguments kernel ----------------------------------------- #

class ConcatArgsKernel(Cohesive): 

   def __init__(self, nodes, sources, layers):

       self._nodes   = nodes
       self._sources = sources
       self._layers  = layers


   def __eq__(self, other):

       return all((
                   self._nodes   == other._nodes,  
                   self._sources == other._sources, 
                   self._layers  == other._layers, 
                 ))


   def __hash__(self):

       return id(self)


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

       return tuple(self._nodes[adx] for adx in self.adxs())


   @tdutil.cacheable
   def deshell(self):

       args = list(self._nodes)

       for adx in self.adxs():
           args[adx] = self._sources[adx]

       return tuple(args)


   @tdutil.cacheable
   def deshelled(self):

       glue = NodeGlue(self.deshell())

       return glue.concatenate()




# --- Concatenated arguments ------------------------------------------------ #

class ConcatArgs(Cohesive):

   def __init__(self, args):

       self._args = args


   def __eq__(self, other):

       return self._args == other._args


   def __hash__(self):

       return id(self)


   @property
   @tdutil.cacheable
   def _kernel(self):

       return NodeGlue(self._args).concatenate()


   def layer(self):

       return self._kernel.layer()


   def adxs(self):

       return self._kernel.adxs()


   def parents(self):

       return self._kernel.parents()


   def deshell(self):

       return self._kernel.deshell()


   def deshelled(self):

       return self.__class__(self._kernel.deshell())




# --- Packable concatenated arguments --------------------------------------- #

class PackableConcatArgs(Cohesive, Packable):

   def __init__(self, args):

       self._args = args


   def __eq__(self, other):

       return self._args == other._args


   def __hash__(self):

       return id(self)


   def layer(self):

       return self._args.layer()


   def adxs(self):

       return self._args.adxs()


   def parents(self):

       return self._args.parents()


   def deshell(self):

       return self._args.deshell()


   def deshelled(self):

       return self.__class__(self._args.deshelled())




# --- Decorator with the default packing behavior --------------------------- #

def default_pack(fun):

    def wrap(self, *args, **kwargs):

        if not self.adxs():
           return PointPack(self.deshell()) 
                                         
        source = self.deshelled().pack()

        return fun(self, source)

    return wrap




# --- Active concatenated arguments ----------------------------------------- #

class Active(PackableConcatArgs):

   @default_pack
   def pack(self, source):

       logic = tdnode.make_logic(
                                 self.parents(),
                                 self.adxs(), 
                                 source,
                                 self.deshell() 
                                )  
 
       return ActivePack(source, self.layer(), logic)




# --- Passive concatenated arguments ---------------------------------------- #

class Passive(PackableConcatArgs):

   @default_pack
   def pack(self, source):
    
       return PassivePack(source)




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




# --- Active pack ----------------------------------------------------------- #

class ActivePack(Pack):

   def __init__(self, source, layer, logic):

       self._source = source
       self._layer  = layer
       self._logic  = logic


   def __eq__(self, other):

       return all((
                   self._source == other._source,
                   self._layer  == other._layer,
                   self._logic  == other._logic,
                 ))


   def __hash__(self):

       return id(self)


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)

       return tdnode.make_node(
                               source, 
                               self._layer, 
                               fun.gate(self._logic) 
                              )




# --- Passive pack ---------------------------------------------------------- #

class PassivePack(Pack):

   def __init__(self, source):

       self._source = source


   def __eq__(self, other):

       return self._source == other._source


   def __hash__(self):

       return id(self)


   def pluginto(self, fun): 

       return self._source.pluginto(fun)




# --- Point pack ------------------------------------------------------------ #

# TODO Future sol: let Array impl Node interface and act as a Point instead!
# i.e. we'll replace Point with Array. Then Array.tovalue() will return self.


class PointPack(Pack): 

   def __init__(self, nodes):

       self._nodes = nodes


   def __eq__(self, other):

       return self._nodes == other._nodes


   def __hash__(self):

       return id(self)


   def pluginto(self, fun):

       return fun(*(node.tovalue() for node in self._nodes))




