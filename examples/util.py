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




class TensorArgs:

   def __init__(self, ts):

       self._ts = ts


   def _t(self, i):
       
       return self._ts[i] 


   def _size(self, i):

       return self._ts.size(i)


   def _inds(self, i):

       return self._ts.inds(i) 


   def _indflat(self, i):

       return self._ts.indflat(i)


   @property
   def _coords(self):

       return self._ts.coords()


   @property
   def _ts(self):

       return iter(self._ts)

       
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




class Optimize:

   def __init__(self, fun, ts):

       self._fun = fun
       self._ts  = ts


   def _costfun(self):

       def wrap(x):

           ts    = self._ts.unpack(x)
           val   = ts.apply(self._fun)
           grads = ts.gradient(self._fun)

           return td.asdata(val), grads.pack()

       return wrap


   def __call__(self, jac=True, method="L-BFGS-B"):

       result = scipy.optimize.minimize(
                   self._costfun, 
                   self._ts.pack(), 
                   jac=jac, 
                   method=method
                )

       return self._ts.unpack(result.x)






