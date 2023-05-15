#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.tensor.data as data

import tadpole.util     as util
import tadpole.tensor   as tn
import tadpole.index    as tid
import tadpole.autodiff as ad

from tests.common import (
   available_backends,
   arepeat,
   arange,
   amap,
)




###############################################################################
###                                                                         ###
###  Function wrappers                                                      ###
###                                                                         ###
###############################################################################


# --- Checkpointed function wrap -------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestCheckpoint:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_checkpoint(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       def fun(x, y):
           return 2*x + y + 5

       cfun = ad.checkpoint(fun)

       def fun1(x):
           return fun(x, x/3.) + fun(x, x**2)

       def fun2(x):
           return cfun(x, x/3.) + cfun(x, x**2)       

       assert tn.allclose(
                 fun1(x.tensor), 
                 fun2(x.tensor)
              )
       assert tn.allclose(
                 ad.gradient(fun1)(x.tensor), 
                 ad.gradient(fun2)(x.tensor)
              )


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_checkpoint_001(self, indnames, shape):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       def fun(*xs):
           return sum(xs)

       cfun = ad.checkpoint(fun)

       def fun1(x):
           return fun(x, x/3.)

       def fun2(x):
           return cfun(x, x/3.)     

       assert tn.allclose(
                 fun1(x.tensor), 
                 fun2(x.tensor)
              )
       assert tn.allclose(
                 ad.gradient(fun1)(x.tensor), 
                 ad.gradient(fun2)(x.tensor)
              )


   #@pytest.mark.skip
   @pytest.mark.parametrize("indnames, shape", [
      ["i", (100000,)],
   ])
   def test_checkpoint_memory(self, indnames, shape):

       try:
           import memory_profiler as mem
           import guppy
       except ImportError:
           return

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, dtype="float64", seed=1
           )

       def fun(x):
           for _ in range(50): 
               x = tn.sin(x**2 + 1)
           return x

       cfun = ad.checkpoint(fun)

       def fun1(fun, x):
           for _ in range(5):
               x = fun(x)
           return tn.sumover(x)

       gradfun = ad.gradient(fun1, 1)

       background_checkpt = max(mem.memory_usage((lambda v: v, (1,))))
       max_usage_checkpt  = max(mem.memory_usage((gradfun, (cfun, x.tensor))))

       background = max(mem.memory_usage((lambda v: v, (1,))))
       max_usage  = max(mem.memory_usage((gradfun, (fun,  x.tensor))))

       max_usage_checkpt -= background_checkpt  
       max_usage         -= background

       h = guppy.hpy()
       print(h.heap())
       print("\nMEMORY USAGE ")
       print("Checkpointed version: ", max_usage_checkpt) 
       print("Original version:     ", max_usage) 
       print("Ratio:                ", max_usage_checkpt / max_usage)

       assert max_usage_checkpt < (max_usage / 2.)









