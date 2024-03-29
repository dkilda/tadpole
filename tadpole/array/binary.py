#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util

import tadpole.array.backends as backends
import tadpole.array.types    as types
import tadpole.array.unary    as unary
import tadpole.array.nary     as nary




###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for binary functions ---------------------------------------- #

def typecast(fun):

    def isarray(x):

        return isinstance(x, types.Array)


    def wrap(x, y, *args, **kwargs):

        if isarray(x) and isarray(y):
           return fun(x, y, *args, **kwargs)       
 
        if not isarray(x) and not isarray(y):  
           x = unary.asarray(x)
           y = unary.asarray(y) 

        if not isarray(x):
           x = y.new(x) 

        if not isarray(y):
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


   # --- Array methods --- #

   def new(self, data):

       return unary.asarray(data, backend=self._backend) 


   def nary(self):

       return nary.Array(self._backend, *self._datas)


   def __or__(self, other):

       backend = backends.common(
          self._backend, 
          other._backend, 
          msg=f"{type(self).__name__}.__or__"
       )

       return nary.Array(backend, *self._datas, *other._datas)


   # --- Comparison --- #

   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._backend, other._backend)
 
       if bool(log):
          return all(self._backend.allequal(x, y) 
                        for x, y in zip(self._datas, other._datas))
               
       return False


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


   def greater(self):

       data = self._backend.greater(*self._datas)

       return self.new(data)

        
   def less(self):

       data = self._backend.less(*self._datas)

       return self.new(data)


   def greater_equal(self):

       data = self._backend.greater_equal(*self._datas)

       return self.new(data)

        
   def less_equal(self):

       data = self._backend.less_equal(*self._datas)

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


   def mod(self):

       data = self._backend.mod(*self._datas)

       return self.new(data)


   def floordiv(self):

       data = self._backend.floordiv(*self._datas)

       return self.new(data)


   def power(self):

       data = self._backend.power(*self._datas)

       return self.new(data)


   # --- Contraction --- #

   def dot(self):

       data = self._backend.dot(*self._datas)

       return self.new(data)
       

   def kron(self):

       data = self._backend.kron(*self._datas)

       return self.new(data)


   # --- Linear algebra --- #

   def solve(self):

       data = self._backend.solve(*self._datas)

       return self.new(data)


   def trisolve(self, which=None):

       data = self._backend.trisolve(*self._datas, which=which)

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
def greater(x, y):

    return (x | y).greater()


@typecast        
def less(x, y):

    return (x | y).less()


@typecast        
def greater_equal(x, y):

    return (x | y).greater_equal()


@typecast        
def less_equal(x, y):

    return (x | y).less_equal()


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
def mod(x, y):

    return (x | y).mod()


@typecast       
def floordiv(x, y):

    return (x | y).floordiv()


@typecast
def power(x, y):

    return (x | y).power()




# --- Contraction ----------------------------------------------------------- #

@typecast
def dot(x, y):

    return (x | y).dot()

     
@typecast  
def kron(x, y):

    return (x | y).kron()




# --- Linear algebra -------------------------------------------------------- #

def solve(a, b):

    return (a | b).solve()


def trisolve(a, b, which=None):

    return (a | b).trisolve(which=which)




