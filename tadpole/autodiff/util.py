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









































































