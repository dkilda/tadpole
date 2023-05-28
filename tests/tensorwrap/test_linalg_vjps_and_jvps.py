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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_000,
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


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_eig_vjp(self, decomp_input, dtype):

       w = data.eig_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       def fun(x):
           V, S = la.eig(x, sind="s") 
           return tc.container(tn.absolute(V), tn.absolute(S)) 

       opts = {}
       if 'complex' in dtype: 
          opts = {"submode": "real"}

       assert_grad(fun, modes="vjp", **opts)(x)


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
   ])
   def test_eig_jvp(self, decomp_input, dtype):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       def fun(x):
           x    = (x(lind,rind) + x.H(lind,rind)) / 2
           V, S = la.eig(x, sind="s") 
           return tc.container(tn.absolute(V), tn.absolute(S)) 

       assert_grad(fun, modes="jvp")(x) 


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_eigh_vjp(self, decomp_input, dtype):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       if 'complex' in dtype: 
          opts = {"submode": "real"}
          def fun(x):
              V, S = la.eigh(x, sind="s") 
              return tc.container(tn.absolute(V), S)  
       else:
          opts = {}
          def fun(x):
              return la.eigh(x, sind="s")

       assert_grad(fun, modes="vjp", **opts)(x) 


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
   ])
   def test_eigh_jvp(self, decomp_input, dtype):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       def fun(x):
           x = (x(lind,rind) + x.H(lind,rind)) / 2
           return la.eigh(x, sind="s")

       assert_grad(fun, modes="jvp")(x) 


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_000,
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      #"complex128",
   ])
   def test_qr(self, decomp_input, dtype):

       if 'complex' in dtype: 
          opts = {"modes": "vjp", "submode": "real"}
          def fun(x): 
              #return la.qr(x, sind="s")
              Q, R = la.qr(x, sind="s") 
              return tc.container(tn.absolute(Q), tn.absolute(R + 1e-10) - 1e-10) 
       else:
          opts = {}
          def fun(x):
              return la.qr(x, sind="s")

       w = data.qr_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, **opts)(x)  


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_000,
      data.decomp_input_001,
      data.decomp_input_002,
      data.decomp_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      #"complex128",
   ])
   def test_lq(self, decomp_input, dtype):

       if 'complex' in dtype: 
          opts = {"modes": "vjp", "submode": "real"}
          def fun(x): 
              L, Q = la.lq(x, sind="s") 
              return tc.container(tn.absolute(L + 1e-10) - 1e-10, tn.absolute(Q)) 
       else:
          opts = {}
          def fun(x):
              return la.lq(x, sind="s")

       w = data.lq_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, **opts)(x)  




###############################################################################
###                                                                         ###
###  Linalg decomposition grads: auxiliary tests                            ###
###                                                                         ###
###############################################################################


# --- Linalg decomposition grads: auxiliary tests --------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsDecompAuxiliary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Auxiliary tests: svd --- #

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


   # --- Auxiliary tests: eig --- #

   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   def _test_eig_vjp_first_order(self, decomp_input):

       def fun(x):
           V, S = la.eig(x, sind="s")
           return tc.container(tn.absolute(V), tn.absolute(S))

       w = data.eig_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       v = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 

       y = fun(x)
       g = tc.container(
              tn.space(y[0]).randn(), 
              tn.space(y[1]).randn(), 
           )  
       assert_vjp_custom(fun, x, g) 


   # --- Auxiliary tests: eigh --- #

   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   def _test_eigh_vjp_first_order(self, decomp_input):

       def fun(x):
           return la.eigh(x, sind="s")

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       v = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 

       out = tc.ContainerGen(v, s)
       g   = tc.ContainerGen(
                tn.space(v).randn(), 
                tn.space(s).randn(), 
             )  

       assert_vjp_custom(fun, x, g) 


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   def _test_eigh_jvp_first_order(self, decomp_input):

       def fun(x):
           return la.eigh(x, sind="s")

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       v = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 

       g = tn.space(x).ones()
       assert_jvp_custom(fun, x, g) 


   @pytest.mark.parametrize("decomp_input", [
      data.decomp_input_002,
   ])
   def _test_eigh_vjp_second_order(self, decomp_input):

       w = data.eigh_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" 
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexGen("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       v = tn.TensorGen(w.lmatrix, (lind, sind)) 
       s = tn.TensorGen(w.smatrix, (sind,     )) 


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

           out  = la.eigh(x, sind="s")
           v, s = out[0], out[1]

           dv = tn.space(v).ones()
           ds = tn.space(s).ones()

           grad = eye(s,"ij") * ds("i1")

           if not tn.allclose(dv, tn.space(dv).zeros()): 
              grad = grad + fmatrix(s)("ij") * (v.T("im") @ dv("mj"))

           grad = v("li").C @ grad @ v.T("jr") 

           tl   = la.tril(tn.space(grad).ones(), k=-1)
           grad = tn.real(grad) * eye(grad) \
                + (grad("lr") + grad.H("lr")) * tl("lr") 
       
           return grad(*tn.union_inds(x))


       assert_grad(fun, order=1, modes="vjp")(x) 


   # --- Auxiliary tests: qr --- #

   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_000,
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   @pytest.mark.parametrize("dtype", [
      #"float64",
      "complex128",
   ])
   def _test_qr_vjp_first_order(self, decomp_input, dtype):

       if 'complex' in dtype: 
          opts = {} #{"submode": "real"} #{"modes": "vjp", "submode": "real"}
          def fun(x):
              #return la.qr(x, sind="s")
              Q, R = la.qr(x, sind="s") 
              return tn.absolute(Q) #tc.container(tn.absolute(Q), R) #tn.absolute(R + 1e-10) - 1e-10)  
       else:
          opts = {}
          def fun(x):
              return la.qr(x, sind="s")

       w = data.qr_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])
       sind = IndexLit("s", min(w.xmatrix.shape[0], w.xmatrix.shape[1]))

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 
       q = tn.TensorGen(w.lmatrix, (lind, sind)) 
       r = tn.TensorGen(w.rmatrix, (sind, rind)) 

       out = tc.ContainerGen(q, r)

       """
       g   = tc.ContainerGen(
                tn.space(q).ones(), 
                tn.space(r).zeros(), 
             )
       """
         
       g = tn.space(q).ones()  

       assert_vjp_custom(fun, x, g) 




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


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_solve(self, solver_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(a, b):
           return la.solve(a, b)

       w = data.solve_linalg_dat(solver_input)(
              data.randn, self.backend, dtype=dtype
           )

       i = IndexGen("i", w.sizeI)
       j = IndexGen("j", w.sizeJ)
       k = IndexGen("k", w.sizeK)

       A = tn.TensorGen(w.matrixA, (i, j)) 
       B = tn.TensorGen(w.matrixB, (i, k)) 

       assert_grad(fun, **opts)(A, B) 


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_trisolve(self, solver_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(a, b):
           return la.trisolve(a, b)

       w = data.trisolve_upper_linalg_dat(solver_input)(
              data.randn, self.backend, dtype=dtype
           )

       i = IndexGen("i", w.sizeI)
       j = IndexGen("j", w.sizeJ)
       k = IndexGen("k", w.sizeK)

       A = tn.TensorGen(w.matrixA, (i, j)) 
       B = tn.TensorGen(w.matrixB, (i, k)) 

       assert_grad(fun, **opts)(A, B) 


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_trisolve_upper(self, solver_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(a, b):
           return la.trisolve(a, b, which="upper") 

       w = data.trisolve_upper_linalg_dat(solver_input)(
              data.randn, self.backend, dtype=dtype
           )

       i = IndexGen("i", w.sizeI)
       j = IndexGen("j", w.sizeJ)
       k = IndexGen("k", w.sizeK)

       A = tn.TensorGen(w.matrixA, (i, j)) 
       B = tn.TensorGen(w.matrixB, (i, k)) 

       assert_grad(fun, **opts)(A, B) 


   @pytest.mark.parametrize("solver_input", [
      data.solver_input_000, 
      data.solver_input_001, 
      data.solver_input_002,
   ])
   @pytest.mark.parametrize("dtype", [
      "float64",
      "complex128",
   ])
   def test_trisolve_lower(self, solver_input, dtype):

       opts = {"modes": "vjp"} if 'complex' in dtype else {}         

       def fun(a, b):
           return la.trisolve(a, b, which="lower") 

       w = data.trisolve_lower_linalg_dat(solver_input)(
              data.randn, self.backend, dtype=dtype
           )

       i = IndexGen("i", w.sizeI)
       j = IndexGen("j", w.sizeJ)
       k = IndexGen("k", w.sizeK)

       A = tn.TensorGen(w.matrixA, (i, j)) 
       B = tn.TensorGen(w.matrixB, (i, k)) 

       assert_grad(fun, **opts)(A, B) 




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


   @pytest.mark.parametrize("shapes, inds, outshape, outinds, which", [
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (8,4),  "lr", None   ], 
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (8,4),  "lr", "left" ], 
      [[(4,4), (4,4)       ], ["ij", "ij"      ], (4,8),  "lr", "right"],
      [[(4,4), (5,4)       ], ["ij", "kj"      ], (9,4),  "lr", None   ],  
      [[(4,4), (5,4)       ], ["ij", "kj"      ], (9,4),  "lr", "left" ], 
      [[(4,4), (4,5)       ], ["ij", "ik"      ], (4,9),  "lr", "right"], 
      [[(4,4), (5,4), (6,4)], ["ij", "kj", "lj"], (15,4), "lr", None   ], 
      [[(4,4), (5,4), (6,4)], ["ij", "kj", "lj"], (15,4), "lr", "left" ], 
      [[(4,4), (4,5), (4,6)], ["ij", "ik", "il"], (4,15), "lr", "right"], 
   ])  
   def test_concat(self, shapes, inds, outshape, outinds, which):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )
       v = data.indices_dat(outinds, outshape)   

       def fun(*xs):
           return la.concat(*xs, inds=v.inds, which=which)

       for i in range(len(w.tensors)):
           assert_grad(fun, i)(*w.tensors) 




