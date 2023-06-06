#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
   IndexLit,
)

import scipy.optimize




def tprint(msg, expr):

    print(msg, td.asdata(expr))




class TensorCollection:

   def __init__(self, tensors):

       self._tensors = tensors


   def __getitem__(self, i):
       
       return self._tensors[i] 


   def __iter__(self):

       for i in self.coords():
           yield self._tensors[i]


   def __len__(self):

       return len(self._tensors)


   def size(self, i):

       return self._tensors[i].size


   def inds(self, i):

       return tuple(td.union_inds(self._tensors[i]))


   def indflat(self, i):

       return f"flat{i}"


   def coords(self):

       return iter(self._tensors.keys())




class ArgsTensorCollection:

   def __init__(self, tensorcollection):

       self._tensors = tensorcollection


   def _t(self, i):
       
       return self._tensors[i] 


   def _size(self, i):

       return self._tensors.size(i)


   def _inds(self, i):

       return self._tensors.inds(i) 


   def _indflat(self, i):

       return self._tensors.indflat(i)


   @property
   def _coords(self):

       return self._tensors.coords()


   @property
   def _ts(self):

       return iter(self._tensors)


   def astensorcollection(self):

       return self._tensors

       
   def pack(self):

       data = []

       for i in self._coords: 

           t = td.fuse(self._t(i), {self._inds(i): self._indflat(i)})
        
           data.append(td.asdata(t, backend="numpy").reshape(-1))

       return np.concatenate(tuple(data))


   def unpack(self, vec):

       ts    = {}
       start = 0

       for i in self._coords: 

           t = td.astensor(
                  vec[start : start + self._size(i)], 
                  (self._indflat(i), )
               )
           t = td.split(t, {self._indflat(i): self._inds(i)})

           start += self._size(i)
           ts[i]  = td.transpose_like(t, self._t(i))

       return self.__class__(ts)


   def apply(self, fun):

       return fun(*self._ts)


   def gradient(self, fun):

       grads = {i: td.gradient(fun, i)(*self._ts) for i in self._coords}

       return self.__class__(grads)


   def evaluate_with_gradient(self, fun):

       return self.apply(fun), self.gradient(fun)




class ArgsTensor:

   def __init__(self, t):

       self._t = t


   @property
   def _inds(self):

       return tuple(td.union_inds(self._t)


   @property
   def _indflat(self):

       return "flat"


   def astensor(self):

       return self._t

       
   def pack(self):

       t = td.fuse(self._t, {self._inds: self._indflat})
        
       return td.asdata(t, backend="numpy").reshape(-1)
 

   def unpack(self, vec):

       t = td.astensor(vec, (self._indflat, ))
       t = td.split(t, {self._indflat: self._inds})

       return self.__class__(td.transpose_like(t, self._t))


   def apply(self, fun):

       return fun(self._t)


   def gradient(self, fun):

       return self.__class__(td.gradient(fun)(self._t))


   def evaluate_with_gradient(self, fun):

       val, grad = td.evaluate_with_gradient(fun)(self._t)

       return val, self.__class__(grad)



class Optimize:

   def __init__(self, fun, ts):

       self._fun = fun
       self._ts  = ts


   def _costfun(self):

       def wrap(x):

           ts    = self._ts.unpack(x)
           val   = ts.apply(self._fun)
           grads = ts.gradient(self._fun)

           return td.asdata(val, backend="numpy"), grads.pack()

       return wrap


   def __call__(self, jac=True, method="L-BFGS-B", options=None):

       result = scipy.optimize.minimize(
                   self._costfun, 
                   self._ts.pack(), 
                   jac=jac, 
                   method=method,
                   options=options
                )

       return self._ts.unpack(result.x)






