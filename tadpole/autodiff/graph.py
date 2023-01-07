#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

from tadpole.autodiff.node import make_node

from tadpole.autodiff.util import cacheable
from tadpole.autodiff.util import Stack


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




# --- Glue interface -------------------------------------------------------- #

class Glue(abc.ABC):

   @abc.abstractmethod
   def pack(self, fun):
       pass




# --- Node glue ------------------------------------------------------------- #

class NodeGlue(Glue):

   def __init__(self, sources, gates):

       self._sources = sources
       self._gates   = gates


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


   def pluginto(self, fun): 

       source = self._source.pluginto(fun)
       gate   = fun.gate(self._inputs)

       return make_node(source, gate, self._layer)



# --- Point pack ------------------------------------------------------------ #

class PointPack(Pack):

   def __init__(self, source):
       self._source = source

   def pluginto(self, fun):
       return self._source.pluginto(fun)

      


# --- Empty pack ------------------------------------------------------------ #

class EmptyPack(Pack):

   def __init__(self, funcall):
       self._funcall = funcall

   def pluginto(self, fun):
       return self._funcall.execute(fun)




# --- Function call --------------------------------------------------------- #

class FunCall:

   def __init__(self, args=None):

       if args is None:
          args = Stack()

       self._args = args


   def add(self, *args):

       xs = self._args

       for arg in args:
           xs = xs.push(arg)

       return self.__class__(xs)


   def execute(self, fun):

       return fun(*self._args.riter())


























































