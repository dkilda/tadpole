#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util      as util
import tadpole.autodiff  as ad
import tadpole.array     as ar
import tadpole.container as tc
import tadpole.tensor    as tn
import tadpole.index     as tid

import tadpole.linalg.unwrapped as la
import tadpole.autodiff.grad    as agrad

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data
import tests.array.data       as ardata


from tests.common import (
   available_backends,
)

from tests.tensorwrap.util import (
   assert_grad,
   assert_vjp_decomp,
   assert_vjp_custom,
)

from tadpole.index import (
   Index,
   IndexGen, 
   IndexLit, 
   Indices,
)




###############################################################################
###                                                                         ###
###  Linalg decomposition grads                                             ###
###                                                                         ###
###############################################################################


# --- Linalg decomposition grads -------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsDecomp:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd_ascontainer(self, decomp_input):

       def fun(x, sind): 
           #U, S = tc.ascontainer(x, x)  # U, S, VH = tc.ascontainer(x, x, x) 
            
           out = tc.ascontainer(x, x)

           U = out[0]
           S = out[1]

           return tc.ascontainer(tn.sin(U), tn.sin(S)) #, tn.sin(U)) #tn.absolute(U)) #tn.absolute(U), S, tn.absolute(VH)) 

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, order=2, modes="vjp", submode="decomp")(x, sind="s")     
       #assert False


   #@pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_svd(self, decomp_input, dtype):

       if 'complex' in dtype: 
          def fun(x, sind):
              U, S, VH, error = la.svd(x, sind) 
              return tc.ascontainer(tn.absolute(U), tn.absolute(S), tn.absolute(VH)) 
       else:
          def fun(x, sind):
              return la.svd(x, sind)

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, modes="vjp", submode="decomp")(x, sind="s")     


   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd_grad(self, decomp_input):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       u = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 
       v = tn.TensorGen(w.rmatrix, (sind, rind)) 


       def eye(x, inds=None): 

           if inds is None:
              return tn.space(x).eye()

           sind  = IndexLit(inds[0], x.shape[0])
           sind1 = IndexLit(inds[1], x.shape[-1])

           return tn.space(x).eye(sind, sind1)


       def fmatrix(s): 

           seye = eye(s,"ij")

           return 1. / (s("1j") - s("i1") + seye) - seye 


       def fun(x):

           out     = la.svd(x, sind="s")

           #U, S, VH, error = la.svd(x, sind="s")
           #out = tc.ascontainer(tn.absolute(U), S, tn.absolute(VH))

           u, s, v = out[0], out[1], out[2].H

           du = tn.space(u).ones()
           ds = tn.space(s).ones()
           dv = tn.space(v).ones()

           f = fmatrix(s**2)("ij")

           uTdu = u.T("im") @ du("mj")
           vTdv = v.T("im") @ dv("mj")

           grad = eye(s,"ij") * ds("i1") 
           grad = grad + f * s("1j") * (uTdu("ij") - uTdu.H("ij"))  
           grad = grad + f * s("i1") * (vTdv("ij") - vTdv.H("ij"))

           grad = u("li").C @ grad("ij") @ v.T("jr")  

           return grad(*tn.union_inds(x))


       assert_grad(fun, order=1, modes="vjp", submode="decomp")(x) 
       #assert False


   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd_grad_formula(self, decomp_input):

       def fun(x):
           return la.svd(x, sind="s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       u = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 
       v = tn.TensorGen(w.rmatrix, (sind, rind)) 

       out = tc.ContainerGen(u, s, v)
       dx  = tn.space(x).randn()
       dy  = tn.space(out).randn()

       assert_vjp_custom(fun, x, dx, dy)  


   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd_second_order(self, decomp_input):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       u = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 
       v = tn.TensorGen(w.rmatrix, (sind, rind)) 

       out = tc.ContainerGen(u, s, v)
       # dy  = tn.space(out).randn()
       # dy  = tn.space(out).zeros()
       
       dy = tc.ContainerGen(
               tn.space(u).zeros(), 
               tn.space(s).randn(), 
               tn.space(v).zeros()
            )

       def fun(x):
           return la.svd(x, sind="s")

       def gradfun(x):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(dy)

       assert_vjp_decomp(gradfun, x)  




