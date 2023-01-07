#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# --- Caching for methods with one-time evaluation -------------------------- # 

class CacheFun:

   def __init__(self):
       self._x = None

   def __call__(self, fun, *args, **kwargs):

       if self._x is None:
          self._x = fun(*args, **kwargs)

       return self._x




def cacheable(fun):

    cachefun = CacheFun()

    def funwrap(*args, **kwargs):
        return cachefun(fun, *args, **kwargs)

    return funwrap




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

































































