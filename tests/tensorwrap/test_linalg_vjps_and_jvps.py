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
import tadpole.autodiff.node    as anode
import tadpole.autodiff.nary    as nary

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data
import tests.array.data       as ardata


from tests.common import (
   available_backends,
)

from tests.tensorwrap.util import (
   assert_grad,
   assert_vjp,
   assert_vjp_null,
   assert_vjp_custom,
   assert_jvp,
   assert_jvp_null,
   assert_jvp_custom,
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


   # --- Main tests --- #

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
          opts = {"modes": "vjp", "submode": "real"}
          def fun(x):
              U, S, VH, error = la.svd(x, sind="s") 
              return tc.container(tn.absolute(U), S, tn.absolute(VH))  
       else:
          opts = {}
          def fun(x):
              return la.svd(x, sind="s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, **opts)(x)     


   # --- Auxiliary tests --- #

   #@pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   def _test_svd_vjp_first_order(self, decomp_input):

       def fun(x):
           return la.svd(x, sind="s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       u = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 
       v = tn.TensorGen(w.rmatrix, (sind, rind)) 

       out = tc.ContainerGen(u, s, v)
       g   = tc.ContainerGen(
                tn.space(u).randn(), 
                tn.space(s).randn(), 
                tn.space(v).randn()
             )  
       # g = tn.space(out).randn()

       assert_vjp_custom(fun, x, g)  


   #@pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   def _test_svd_jvp_first_order(self, decomp_input):

       def fun(x):
           U, S, VH, error = la.svd(x, sind="s") 
           return tc.container(tn.absolute(U), S, tn.absolute(VH)) 
           # return la.svd(x, sind="s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="complex128"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       u = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 
       v = tn.TensorGen(w.rmatrix, (sind, rind)) 

       g = tn.space(x).randn()
       assert_jvp_custom(fun, x, g)  

 
   #@pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def _test_svd_vjp_second_order(self, decomp_input):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" 
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


       assert_grad(fun, order=1, modes="vjp")(x) 

 
   #@pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   def _test_svd_jvp_second_order(self, decomp_input):

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" 
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
           u, s, v = out[0], out[1], out[2].H  

           dx = tn.space(x).ones()
           f  = fmatrix(s**2)("ij")

           grad1 = u.H("il") @ dx("lr")   @ v("rj")
           grad2 = v.H("ir") @ dx.H("rl") @ u("lj") 

           ds = 0.5 * eye(s,"ij") * (grad1 + grad2)
           du = u("li") @ (f * (grad1 * s("1j") + s("i1") * grad2))
           dv = v("ri") @ (f * (grad1 * s("i1") + s("1j") * grad2))


           if x.shape[0] < x.shape[1]:

              vvH = v("rm") @ v.H("ma")
              dv  = dv("ri") \
                  + (eye(vvH) - vvH) @ dx.H("ab") @ (u("bi") / s("1i"))


           if x.shape[0] > x.shape[1]:

              uuH = u("lm") @ u.H("ma")
              du  = du("li") \
                  + (eye(uuH) - uuH) @ dx("ab") @ (v("bi") / s("1i")) 


           du = du(*tn.union_inds(u))
           dv = dv(*tn.union_inds(v))
           ds = la.diag(tn.astype_like(ds, s), tuple(tn.union_inds(s)))

           return tc.container(du, ds, dv.H)


       assert_grad(fun, order=1, modes="jvp", submode="real")(x) 




###############################################################################
###                                                                         ###
###  Linalg property grads                                                  ###
###                                                                         ###
###############################################################################


# --- Linalg property grads ------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsProperties:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
      data.property_input_003,
   ])
   @pytest.mark.parametrize("order", [
      None, "fro", "nuc", 
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_norm(self, property_input, order, dtype):

       opts = {}
       if 'complex' in dtype:
          opts = {"modes": "vjp", "submode": "real"} 

       def fun(x):
           return la.norm(x, order=order)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x)  


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
      data.property_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_trace(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x):
           return la.trace(x)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x) 


   @pytest.mark.parametrize("property_input", [
      data.property_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_det(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x):
           return la.det(x)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x) 


   @pytest.mark.parametrize("property_input", [
      data.property_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_inv(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x):
           return la.inv(x)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x) 


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
      data.property_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_tril(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x):
           return la.tril(x)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x) 


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
      data.property_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_triu(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x):
           return la.triu(x)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x) 


   @pytest.mark.parametrize("property_input", [
      data.property_input_001,
      data.property_input_002,
      data.property_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_diag(self, property_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(x, inds):
           return la.diag(x, inds)

       w = data.property_linalg_dat(property_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.matrix.shape[0])
       rind = IndexGen("r", w.matrix.shape[1])
       sind = IndexGen("s", min(w.lsize, w.rsize))

       x = tn.TensorGen(w.matrix, (lind, rind)) 

       assert_grad(fun, **opts)(x, sind) 




###############################################################################
###                                                                         ###
###  Linalg solver grads                                                    ###
###                                                                         ###
###############################################################################


# --- Linalg solver grads --------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsSolvers:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend




###############################################################################
###                                                                         ###
###  Linalg transformation grads                                            ###
###                                                                         ###
###############################################################################


# --- Linalg transformation grads ------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsTransforms:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend




































