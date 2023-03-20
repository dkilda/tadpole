#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad





# IndexPartition constructs [linds/rinds funs] w/o actually computing [linds/rinds] themselves 

class LeftIndexPartition(IndexPartition):

   def left(self, inds): # TODO returns PartialIndices with left-config




class RightIndexPartition(IndexPartition):

   def right(self, inds): # TODO returns PartialIndices with right-config




class PartialIndices: # FIXME impl all (?) of Indices iface?

   







# --- Decomposition call ---------------------------------------------------- #

class DecompCall(FunCall):

   def __init__(self, engine, outinds=None):

       if not isinstance(engine, Engine):
          engine = Engine(engine)

       self._engine  = engine
       self._outinds = outinds


   def attach(self, backend, data, inds):

       return self.__class__(self._engine.attach(backend, data, inds))


   @util.cacheable
   def outinds(self):

       if self._outinds is None: 
          return contract.make_output_inds(inds)

       return self._outinds


   def execute(self):

       backend = next(self._engine.backends())
       datas   = self._engine.datas()
       inds    = self._engine.inds()

       equation = contract.make_einsum_equation(inds, self.outinds())
       outdata  = self._engine.execute(backend, equation, *datas)

       return core.Tensor(backend, outdata, self.outinds() 








# --- Linear algebra: decomposition methods --------------------------------- #

@ad.differentiable
def svd(x, mind, linds=None, rinds=None, **opts):

    def fun(backend, v):
        return backend.svd(v)

    return Args(x).pluginto(SplitCall(fun))



@ad.differentiable
def qr(x):

    def fun(backend, v):
        return backend.qr(v)

    return Args(x).pluginto(DoubleDecompCall(fun))



@ad.differentiable
def eig(x):

    def fun(backend, v):
        return backend.eig(v)

    return Args(x).pluginto(DoubleDecompCall(fun))



@ad.differentiable
def eigh(x):

    def fun(backend, v):
        return backend.eigh(v)

    return Args(x).pluginto(DoubleDecompCall(fun))
       



























































