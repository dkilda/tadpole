#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.misc  as misc




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable function type ------------------------------------------ #

class DifferentiableFun(at.DifferentiableFun):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def __call__(self, *args, **kwargs):

       return self._fun["call", fake.Value()](*args, **kwargs)


   def vjp(self, *args, **kwargs):

       return self._fun["vjp", lambda g: fake.Value()](*args, **kwargs)


   def jvp(self, *args, **kwargs):

       return self._fun["jvp", lambda g: fake.Value()](*args, **kwargs)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

class Args(at.Args):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   @property
   def _items(self):

       return self._fun["items", tuple()]()


   def __eq__(self, other):

       return id(self) == id(other)


   def __hash__(self):

       return id(self)


   def __len__(self):

       return len(self._items)


   def __contains__(self, x):

       return x in self._items


   def __iter__(self):

       return iter(self._items)


   def __getitem__(self, idx):

       return self._items[idx] 


   def nodify(self):

       return self._fun["nodify", arepeat(fake.Node, 2)]()

 
   def concat(self):

       return self._fun["concat", Sequential()]()

 
   def pack(self):

       return self._fun["pack", Pack()]()

 
   def deshelled(self):

       return self._fun["deshelled", self.__class__()]()




# --- Sequential type ------------------------------------------------------- #

class Sequential(at.Sequential):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def attach(self, node, source, layer):

       return self._fun["attach", self.__class__()](node, source, layer)




# --- Cohesive type --------------------------------------------------------- #

class Cohesive(at.Cohesive):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def innermost(self):

       return self._fun["innermost", self.layer() == misc.minlayer()]()


   def layer(self):

       return self._fun["layer", fake.Value()]()

       
   def adxs(self):

       return self._fun["adxs", fake.Value()]()


   def parents(self):

       return self._fun["parents", fake.Parents()]()


   def deshell(self):

       return self._fun["deshell", Args()]()




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Pack type ------------------------------------------------------------- #

class Pack(at.Pack):

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def innermost(self):

       return self._fun["innermost", False]()


   def deshell(self):

       return self._fun["deshell", Args()]()


   def deshelled(self):

       return self._fun["deshelled", self.__class__()]()


   def fold(self, funwrap, out):

       return self._fun["fold", fake.Node()](funwrap, out)




# --- Envelope type --------------------------------------------------------- #

class Envelope(at.Envelope): 

   def __init__(self, **data):  

       self._fun = fake.FunMap(**data)


   def packs(self):

       default = util.Loop(
                           Pack(), 
                           lambda x: Pack(), 
                           lambda x: True
                          )

       return self._fun["packs", default]()


   def apply(self, fun):

       return self._fun["apply", fake.Value()](fun)


   def applywrap(self, funwrap, fun):

       return self._fun["applywrap", fake.Node()](funwrap, fun)




