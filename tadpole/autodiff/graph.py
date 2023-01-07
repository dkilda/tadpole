#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.autodiff.node import make_node
from tadpole.autodiff.util import cacheable




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




# --- Node glue ------------------------------------------------------------- #

class NodeGlue:

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

class PointGlue:

   def __init__(self, sources):

       self._sources = sources


   def pack(self, funcall):

       source = ArgGlue(self._sources.args()).pointpack(funcall)

       return PointPack(source, self._sources.layer())




































































