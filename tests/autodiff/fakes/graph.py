#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph

from tests.common.fakes        import NULL, fakeit
from tests.autodiff.fakes.misc import Fun, FunReturn, Map




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class Graph:

   def __init__(self, build=NULL):

       self._entered = False
       self._exited  = False
       self._build   = build


   def entered(self):

       return self._entered


   def exited(self):

       return self._exited


   def __enter__(self):

       self._entered = True
       self._exited  = False
       return self


   def __exit__(self, exception_type, exception_val, trace):

       self._entered = False
       self._exited  = True


   @fakeit
   def build(self, gate):

       return self._build[gate]




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Function with gate ---------------------------------------------------- #

class FunWithGate:

   def __init__(self, call=NULL, gate=NULL):

       if call is NULL:
          call = Map(FunReturn())

       self._call = call
       self._gate = gate


   @fakeit
   def __call__(self, *args):

       return self._call[args]


   @fakeit
   def gate(self, logic):

       return self._gate[logic]




# --- Differentiable function decorator ------------------------------------- #

Differentiable = Fun




# --- Non-differentiable function decorator --------------------------------- #

NonDifferentiable = Fun




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

class NodeTrain:

   def __init__(self, with_node=NULL, with_meta=NULL, concatenate=NULL):

       self._with_node   = with_node
       self._with_meta   = with_meta
       self._concatenate = concatenate 


   @fakeit
   def with_node(self, node):

       return self._with_node


   @fakeit
   def with_meta(self, source, layer):

       return self._with_meta


   @fakeit
   def concatenate(self):

       return self._concatenate 




# --- Node glue ------------------------------------------------------------- #

class NodeGlue:

   def __init__(self, iterate=NULL, concatenate=NULL):

       self._iterate     = iterate
       self._concatenate = concatenate 


   @fakeit
   def iterate(self):
 
       return self._iterate


   @fakeit
   def concatenate(self):

       return self._concatenate




###############################################################################
###                                                                         ###
###  Concatenated arguments                                                 ###
###                                                                         ###
###############################################################################


# --- Concatenated arguments ------------------------------------------------ #

class ConcatArgs(tdgraph.Cohesive): 

   def __init__(self, layer=NULL, adxs=NULL, parents=NULL, 
                      deshell=NULL, deshelled=NULL):

       self._layer     = layer
       self._adxs      = adxs
       self._parents   = parents
       self._deshell   = deshell
       self._deshelled = deshelled


   @fakeit
   def layer(self):

       return self._layer


   @fakeit
   def adxs(self):

       return self._adxs


   @fakeit
   def parents(self):

       return self._parents


   @fakeit
   def deshell(self):

       return self._deshell


   @fakeit
   def deshelled(self):

       return self._deshelled




# --- Packable concatenated arguments --------------------------------------- #

class PackableConcatArgs(tdgraph.Cohesive, tdgraph.Packable):

   def __init__(self, layer=NULL, adxs=NULL, parents=NULL, 
                      deshell=NULL, deshelled=NULL, pack=NULL):

       self._layer     = layer
       self._adxs      = adxs
       self._parents   = parents
       self._deshell   = deshell
       self._deshelled = deshelled
       self._pack      = pack


   @fakeit
   def layer(self):

       return self._layer


   @fakeit
   def adxs(self):

       return self._adxs


   @fakeit
   def parents(self):

       return self._parents


   @fakeit
   def deshell(self):

       return self._deshell


   @fakeit
   def deshelled(self):

       return self._deshelled


   @fakeit
   def pack(self):

       return self._pack




# --- Active concatenated arguments ----------------------------------------- #

Active = PackableConcatArgs




# --- Passive concatenated arguments ---------------------------------------- #

Passive = PackableConcatArgs




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Pack ------------------------------------------------------------------ #

class Pack(tdgraph.Pack):

   def __init__(self, pluginto=NULL):

       self._pluginto = pluginto


   def pluginto(self, fun):

       return self._pluginto[fun]




# --- Active pack ----------------------------------------------------------- #

ActivePack = Pack




# --- Passive pack ---------------------------------------------------------- #

PassivePack = Pack




# --- Point pack ------------------------------------------------------------ #

PointPack = Pack




