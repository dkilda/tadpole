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
   assert_vjp_container,
   assert_vjp_decomp,
   assert_vjp_custom,
   assert_jvp,
   assert_jvp_null,
   assert_jvp_container,
   assert_jvp_decomp,
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


   # --- ADVANCED --- #

   @pytest.mark.parametrize("indnames, shape, squeezed", [
      ["iajkbcl", (2,1,3,4,1,1,5), None],
      ["iajkbcl", (2,1,3,4,1,1,5), "ac"],
   ])
   def test_gradfun_squeeze(self, indnames, shape, squeezed):

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       def fun(x, squeezed):
           return tn.squeeze(x, squeezed) 

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

           def gradfun(x, g):
               op = agrad.diffop_reverse(fun, x)
               return op.grad(g)

           g = tn.space(fun(x)).randn()

           _assert_grad_1(gradfun, (0,1))(x, g)

       @nary.nary_op
       def _assert_grad_1(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

       _assert_grad(fun)(x.tensor, squeezed)


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_gradfun_where(self, indnames, shape, nvals):

       w = data.tensor_dat(data.randn_pos)(
              self.backend, 
              indnames, 
              shape, 
              seed=1, 
              dtype="bool", 
              nvals=nvals, 
              defaultval=False
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=2
           )
       y = data.array_dat(data.randn)(
              self.backend, w.shape, seed=3
           )

       wtensor = w.tensor
       xtensor = tn.TensorGen(x.array, w.inds)
       ytensor = tn.TensorGen(y.array, w.inds)

       def fun(v, x, y):
           return tn.where(v, x, y) 

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_null(fun, x)
           #assert_vjp_container(fun, x)

           def gradfun(x, g):
               op = agrad.diffop_reverse(fun, x)
               return op.grad(g)

           g = tn.space(fun(x)).randn()

           _assert_grad_1(gradfun, (0,1))(x, g)

       @nary.nary_op
       def _assert_grad_1(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_null(fun, x)
           #assert_vjp_container(fun, x)

       _assert_grad(fun, 0)(wtensor, xtensor, ytensor)
       #_assert_grad(fun, 1)(wtensor, xtensor, ytensor)
       #_assert_grad(fun, 2)(wtensor, xtensor, ytensor)


   @pytest.mark.skip
   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_gradfun_000(self, indnames, shape):

       def fun(x):
           U, S = tc.container(x,x) 
           return tc.container(U, tn.sin(S)) 

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       g = tn.space(fun(x.tensor)).randn()

       _assert_grad(gradfun, (0,1))(x.tensor, g)
       #_assert_grad(fun)(x.tensor)


   # --- VJP's --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_gradfun_001(self, indnames, shape):

       def fun(x):
           return tn.sin(x) #tn.absolute(x) #tn.sin(x)

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       g = tn.space(fun(x.tensor)).randn()

       _assert_grad(gradfun, (0,1))(x.tensor, g)
       #_assert_grad(fun)(x.tensor)


   @pytest.mark.parametrize("indnames, shape, squeezed", [
      ["iajkbcl", (2,1,3,4,1,1,5), None],
      ["iajkbcl", (2,1,3,4,1,1,5), "ac"],
   ])
   def test_gradfun_002(self, indnames, shape, squeezed):

       def fun(x):
           return tn.squeeze(x, squeezed) 

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       g = tn.space(fun(x.tensor)).randn()

       _assert_grad(gradfun, (0,1))(x.tensor, g)
       #_assert_grad(fun)(x.tensor)


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_gradfun_003(self, indnames, shape, nvals):

       w = data.tensor_dat(data.randn_pos)(
              self.backend, 
              indnames, 
              shape, 
              seed=1, 
              dtype="bool", 
              nvals=nvals, 
              defaultval=False
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=2
           )
       y = data.array_dat(data.randn)(
              self.backend, w.shape, seed=3
           )

       wtensor = w.tensor
       xtensor = tn.TensorGen(x.array, w.inds)
       ytensor = tn.TensorGen(y.array, w.inds)

       def fun(v):
           return tn.where(v, xtensor, ytensor)

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_vjp_container(fun, x)

       g = tn.space(fun(w.tensor)).randn()

       _assert_grad(gradfun, (0,1))(wtensor, g)
       #_assert_grad(fun)(x.tensor)


   # --- JVP's --- #

   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   def test_gradfun_004(self, indnames, shape):

       def fun(x):
           return tn.sin(x) #tn.absolute(x) #tn.sin(x)

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_jvp_container(fun, x)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       g = tn.space(fun(x.tensor)).randn()

       _assert_grad(gradfun, (0,1))(x.tensor, g)
       #_assert_grad(fun)(x.tensor)


   @pytest.mark.parametrize("indnames, shape, squeezed", [
      ["iajkbcl", (2,1,3,4,1,1,5), None],
      ["iajkbcl", (2,1,3,4,1,1,5), "ac"],
   ])
   def test_gradfun_005(self, indnames, shape, squeezed):

       def fun(x):
           return tn.squeeze(x, squeezed) 

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_jvp_container(fun, x)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       g = tn.space(fun(x.tensor)).randn()

       _assert_grad(gradfun, (0,1))(x.tensor, g)
       #_assert_grad(fun)(x.tensor)


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_gradfun_006(self, indnames, shape, nvals):

       w = data.tensor_dat(data.randn_pos)(
              self.backend, 
              indnames, 
              shape, 
              seed=1, 
              dtype="bool", 
              nvals=nvals, 
              defaultval=False
           )
       x = data.array_dat(data.randn)(
              self.backend, w.shape, seed=2
           )
       y = data.array_dat(data.randn)(
              self.backend, w.shape, seed=3
           )

       wtensor = w.tensor
       xtensor = tn.TensorGen(x.array, w.inds)
       ytensor = tn.TensorGen(y.array, w.inds)

       def fun(v):
           return tn.where(v, xtensor, ytensor)

       def gradfun(x, g):
           op = agrad.diffop_reverse(fun, x)
           return op.grad(g)

       @nary.nary_op
       def _assert_grad(fun, x):

           if isinstance(x, tuple):
              x = tc.container(x)

           assert_jvp_container(fun, x)

       g = tn.space(fun(w.tensor)).randn()

       _assert_grad(gradfun, (0,1))(wtensor, g)
       #_assert_grad(fun)(x.tensor)


   # --- MAIN --- #

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
          def fun(x):
              U, S, VH, error = la.svd(x, sind="s") 
              return tc.container(tn.absolute(U), tn.absolute(S), tn.absolute(VH)) 
       else:
          def fun(x):
              return la.svd(x, sind="s")

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype=dtype
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, order=2, modes="vjp", submode="decomp")(x)     


   # --- Currently inactive tests --- #

   @pytest.mark.skip
   @pytest.mark.parametrize("decomp_input", [
      #data.decomp_input_001,
      data.decomp_input_002,
      #data.decomp_input_003,
   ])
   def test_svd_ascontainer(self, decomp_input):

       def fun(x, sind): 
           #U, S = tc.container(x, x)  # U, S, VH = tc.container(x, x, x) 
            
           out = tc.container(x, x)

           U = out[0]
           S = out[1]

           return tc.container(tn.sin(U), tn.sin(S)) #, tn.sin(U)) #tn.absolute(U)) #tn.absolute(U), S, tn.absolute(VH)) 

       w = data.svd_tensor_dat(decomp_input)(
              data.randn, self.backend, dtype="float64" #"complex128" #"float64"
           )

       lind = IndexGen("l", w.xmatrix.shape[0])
       rind = IndexGen("r", w.xmatrix.shape[1])

       x = tn.TensorGen(w.xmatrix, (lind, rind)) 

       assert_grad(fun, order=2, modes="vjp", submode="decomp")(x, sind="s")     
       #assert False


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
           #out = tc.container(tn.absolute(U), S, tn.absolute(VH))

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




