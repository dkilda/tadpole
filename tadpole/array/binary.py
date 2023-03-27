#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.backends as backends

import tadpole.array.types as types
import tadpole.array.unary as unary
import tadpole.array.nary  as nary




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for binary functions ---------------------------------------- #

def typecast(fun):

    def wrap(x, y, *args, **kwargs):

        try:
            return fun(x, y, *args, **kwargs)       
 
        except (AttributeError, TypeError):

            if not any(isinstance(v, types.Array) for v in (x,y)):
               x = unary.asarray(x)
               y = unary.asarray(y) 

            if not isinstance(x, types.Array):
               x = y.new(x) 

            if not isinstance(y, types.Array):
               y = x.new(y) 

            return fun(x, y, *args, **kwargs)
         
    return wrap




# --- Approximate (close) equality ------------------------------------------ #

def close_opts(opts):

    rtol = opts.pop("rtol", 1e-5)
    atol = opts.pop("atol", 1e-8)

    return {"rtol": rtol, "atol": atol, **opts}




###############################################################################
###                                                                         ###
###  Definition of Binary Array (supports binary operations)                ###
###                                                                         ###
###############################################################################


# --- Binary Array ---------------------------------------------------------- #

class Array(types.Array):

   # --- Construction --- #

   def __init__(self, backend, dataA, dataB):

       if not isinstance(backend, backends.Backend):
          raise ValueError(
             f"{type(self).__name__}: "
             f"backend must be an instance of Backend, but it is {backend}"
          ) 

       self._backend = backend
       self._datas   = (dataA, dataB)


   # --- Arraylike methods --- #

   def new(self, data):

       return unary.Array(self._backend, data)


   def __or__(self, other):

       backend = backends.common(
          self._backend, 
          other._backend, 
          msg=f"{type(self).__name__}.__or__"
       )

       return nary.Array(backend, *self._datas, *other._datas)


   # --- Logical operations --- #

   def allclose(self, **opts):

       return self._backend.allclose(*self._datas, **close_opts(opts))  


   def allequal(self):

       return self._backend.allequal(*self._datas) 


   def isclose(self, **opts):

       data = self._backend.isclose(*self._datas, **close_opts(opts))  

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

@typecast
def allclose(x, y, **opts):

    return (x | y).allclose(**opts) 


@typecast
def allequal(x, y):

    return (x | y).allequal()


@typecast
def isclose(x, y, **opts):

    return (x | y).isclose()


@typecast
def isequal(x, y):

    return (x | y).isequal()


@typecast
def notequal(x, y):

    return (x | y).notequal()


@typecast
def logical_and(x, y):

    return (x | y).logical_and()


@typecast
def logical_or(x, y):

    return (x | y).logical_or()




# --- Elementwise binary algebra -------------------------------------------- #

@typecast
def add(x, y):

    return (x | y).add()


@typecast
def sub(x, y):

    return (x | y).sub()


@typecast
def mul(x, y):

    return (x | y).mul()


@typecast       
def div(x, y):

    return (x | y).div()


@typecast
def power(x, y):

    return (x | y).power()




# --- Linear algebra: products ---------------------------------------------- #

@typecast
def dot(x, y):

    return (x | y).dot()

     
@typecast  
def kron(x, y):

    return (x | y).kron()




