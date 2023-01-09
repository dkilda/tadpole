#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class MockGraph:

   def __init__(self, build=NULL):

       self._build = build


   def __enter__(self):

       return self


   def __exit__(self, exception_type, exception_val, trace):

       pass


   @mockify
   def build(self, root_gate):

       return self._build = build




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Gated function -------------------------------------------------------- #

class MockGatedFun:

   def __init__(self, call=NULL, gate=NULL):

       self._call = call
       self._gate = gate


   @mockify
   def __call__(self, *args):

       return self._call


   @mockify
   def gate(self, inputs):

       return self._gate




# --- Differentiable and non-differentiable function decorator -------------- #

class MockFun: 

   def __init__(self, call=NULL):

       self._call = call


   @mockify
   def __call__(self, *args):

       return self._call




###############################################################################
###                                                                         ###
###  Argument glue and packs: glueing arguments into a single pack          ###
###                           for function calls.                           ###
###                                                                         ###
###############################################################################


# --- Argument filter ------------------------------------------------------- #

class MockArgFilter:

   def __init__(self, vals=NULL, nodes=NULL):

       self._vals  = vals
       self._nodes = nodes


   @mockify
   def vals(self, args):

       return self._vals


   @mockify
   def nodes(self, args):

       return self._nodes




# --- Glue ------------------------------------------------------------------ #

class MockArgGlue(Glue):

   def __init__(self, pack=NULL):

       self._pack = pack


   @mockify
   def pack(self, funcall=None):

       return self._pack




# --- Function call --------------------------------------------------------- #

class MockFunCall:

   def __init__(self, execute=NULL, next_execute=NULL):

       self._execute      = execute
       self._next_execute = next_execute


   def _next_args(self):

       if self._next_execute is NULL:
          return self._execute

       if len(self._next_execute) == 0:
          return self._execute

       return self._next_execute[0], self._next_execute[1:]


   @mockify
   def add(self, *args):
          
       return self.__class__(*self._next_args()) 


   @mockify
   def execute(self, fun):

       return self._execute[fun]




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node sources ---------------------------------------------------------- #

class MockSources: 

   def __init__(self, layer=NULL, adxs=NULL, args=NULL):

       self._layer = layer
       self._adxs  = adxs
       self._args  = args


   @mockify
   def layer(self):

       return self._layer


   @mockify
   def adxs(self):

       return self._adxs


   @mockify
   def args(self):

       return self._args




# --- Glue ------------------------------------------------------------------ #

class MockGlue(Glue):

   def __init__(self, pack=NULL):

       self._pack = pack


   @mockify
   def pack(self, funcall):

       return self._pack




###############################################################################
###                                                                         ###
###  Node packs: representing multiple nodes by a single argument           ###
###              for function calls.                                        ###
###                                                                         ###
###############################################################################


# --- Pack ------------------------------------------------------------------ #

class MockPack(Pack):

   def __init__(self, pluginto=NULL):

       self._pluginto = pluginto


   @mockify
   def pluginto(self, fun):

       return self._pluginto[fun]





























































