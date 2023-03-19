#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util     as util
import tadpole.backends as backends

import tadpole.array.core  as core
import tadpole.array.unary as unary
import tadpole.array.nary  as nary

from tadpole.array.core import ArrayLike

# TODO TODO TODO need typecast! put it in core!


###############################################################################
###                                                                         ###
###  Definition of Binary Array (supports binary operations)                ###
###                                                                         ###
###############################################################################


# --- Binary Array ---------------------------------------------------------- #

class Array(ArrayLike):

   # --- Construction --- #

   def __init__(self, backend, dataA, dataB):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"TwoArray: backend must be an instance "
             f"of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._datas   = (dataA, dataB)


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.Array(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, other._backend, msg=f"{type(self).__name__}.__or__"
       )

       return unary.Array(backend, *self._datas, *other._datas)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return self._backend.allclose(*self._datas, **opts)  


   def allequal(self):

       return self._backend.allequal(*self._datas) 


   def isclose(self, **opts):

       data = self._backend.isclose(*self._datas, **opts)  

       return self.new(data)
 

   def isequal(self):

       data = self._backend.isequal(*self._datas) 
 
       return self.new(data)


   def notequal(self):

       data = self._backend.notequal(*self._datas)

       return self.new(data)


   def logical_and(self):

       data = self._backend.logical_and(*self._datas)

       return self.new(data)


   def logical_or(self):

       data = self._backend.logical_or(*self._datas)

       return self.new(data)


   # --- Elementwise binary algebra --- #

   def add(self):

       data = self._backend.add(*self._datas)

       return self.new(data)

       
   def sub(self):

       data = self._backend.sub(*self._datas)

       return self.new(data)      


   def mul(self):
       
       data = self._backend.mul(*self._datas)

       return self.new(data)


   def div(self):

       data = self._backend.div(*self._datas)

       return self.new(data)


   def power(self):

       data = self._backend.power(*self._datas)

       return self.new(data)


   # --- Linear algebra: products --- #

   def dot(self):

       data = self._backend.dot(*self._datas)

       return self.new(data)
       

   def kron(self):

       data = self._backend.kron(*self._datas)

       return self.new(data)




###############################################################################
###                                                                         ###
###  Standalone functions corresponding to Binary Array methods             ###
###                                                                         ###
###############################################################################


# --- Logical operations ---------------------------------------------------- #

def allclose(x, y, **opts):

    return (x | y).allclose(**opts) 


def allequal(x, y):

    return (x | y).allequal()


def isclose(x, y, **opts):

    return (x | y).isclose()


def isequal(x, y):

    return (x | y).isequal()


def notequal(x, y):

    return (x | y).notequal()


def logical_and(x, y):

    return (x | y).logical_and()


def logical_or(x, y):

    return (x | y).logical_or()




# --- Elementwise binary algebra -------------------------------------------- #

def add(x, y):

    return (x | y).add()


def sub(x, y):

    return (x | y).sub()


def mul(x, y):

    return (x | y).mul()

       
def div(x, y):

    return (x | y).div()


def power(x, y):

    return (x | y).power()




# --- Linear algebra: products ---------------------------------------------- #

def dot(x, y):

    return (x | y).dot()

       
def kron(x, y):

    return (x | y).kron()




