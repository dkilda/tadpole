#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Cache for methods with one-time evaluation ---------------------------- # 

class CacheFun:

   def __init__(self):
       self._x = None

   def __call__(self, fun, *args, **kwargs):

       if self._x is None:
          self._x = fun(*args, **kwargs)

       return self._x


# --- Decorator for cacheable methods -------------------------------------- #

def cacheable(fun):

    cachefun = CacheFun()

    def funwrap(*args, **kwargs):
        return cachefun(fun, *args, **kwargs)

    return funwrap




###############################################################################
###                                                                         ###
###  A quasi-immutable stack data structure                                 ###
###                                                                         ###
###############################################################################


# --- Stack ----------------------------------------------------------------- #

class Stack:

   def __init__(self, xs=None, end=0):

       if xs is None:
          xs = []

       self._xs  = xs
       self._end = end


   def push(self, x):
       self._xs.append(x)
       return self.__class__(self._xs, self._end + 1)

   def pop(self):
       return self.__class__(self._xs, self._end - 1)

   def top(self):
       return self._xs[self._end - 1] 


   def tolist(self):
       return list(self.riter())

   def iter(self):
       return reversed(self.riter())

   def riter(self):
       for i in range(self.size()):
           yield self._xs[i]


   def size(self):
       return self._end

   def empty(self):
       return self.size() == 0




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy interface ---------------------------------------------- #

class ArgProxy(abc.ABC):

   @abc.abstractmethod
   def insert(self, args, x):
       pass

   @abc.abstractmethod
   def extract(self, args):
       pass




# --- Singular argument proxy (represents a single variable in args) -------- #

class SingularArgProxy(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def insert(self, args, x):
        
       out            = list(args)
       out[self._adx] = x

       return out


   def extract(self, args):

       return args[self._adx]




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class PluralArgProxy(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def insert(self, args, x):

       out = list(args)

       for i, v in zip(self._adx, x):
           out[i] = v  

       return out


   def extract(self, args):

       return tuple(args[i] for i in self._adx)  




# --- Unary function argument proxy ----------------------------------------- # 

def make_argproxy(adx):

    if adx is None:
       adx = 0

    if isinstance(adx, int):
       return SingularArgProxy(adx)

    return PluralArgProxy(adx)


























































