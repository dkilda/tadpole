#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.array.backends as backends

import tadpole.tensorwrap.tensor_vjps as tvjps
import tadpole.tensorwrap.tensor_jvps as tjvps

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data
import tests.array.data       as ardata


from tests.common import (
   available_backends,
)

from tests.tensorwrap.util import (
   assert_grad,
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)

"""
from tadpole.tensorwrap.tensor_vjps.elemwise_unary import (
   sparsegrad,
)
"""



###############################################################################
###                                                                         ###
###  Unary elementwise grads                                                ###
###                                                                         ###
###############################################################################


# --- Unary elementwise grads ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseUnary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   @pytest.mark.parametrize("op", [
      "neg",
      "conj",
      "real",
      "imag",
      "absolute",
      "sqrt",
      "log",
      "exp",
      "sin", 
      "cos",
      "tan",
      "arcsin",
      "arccos",
      "arctan",
      "sinh", 
      "cosh",
      "tanh",
      "arcsinh",
      "arccosh",
      "arctanh",
   ])
   def test_math(self, indnames, shape, op):

       opts = {}
       fun = {
              "neg":      lambda x: -x,
              "conj":     lambda x: tn.conj(x),
              "real":     lambda x: tn.real(x),
              "imag":     lambda x: tn.imag(x),
              "absolute": lambda x: tn.absolute(x),
              "sqrt":     lambda x: tn.sqrt(x),
              "log":      lambda x: tn.log(x),
              "exp":      lambda x: tn.exp(x),  
              "sin":      lambda x: tn.sin(x),
              "cos":      lambda x: tn.cos(x),
              "tan":      lambda x: tn.tan(x),
              "arcsin":   lambda x: tn.arcsin(x),
              "arccos":   lambda x: tn.arccos(x),
              "arctan":   lambda x: tn.arctan(x),
              "sinh":     lambda x: tn.sinh(x), 
              "cosh":     lambda x: tn.cosh(x),
              "tanh":     lambda x: tn.tanh(x),
              "arcsinh":  lambda x: tn.arcsinh(x),
              "arccosh":  lambda x: tn.arccosh(x),
              "arctanh":  lambda x: tn.arctanh(x),
             }[op]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       xtensor = x.tensor
 
       if op == "arccosh":
          xtensor = xtensor + 2.5

       if op in ("conj", "real", "imag"): 
          opts = {"submode": "real"}

       if op in ("absolute",):
          opts = {"order": 3, "submode": "real"}

       assert_grad(fun, **opts)(xtensor)


   @pytest.mark.filterwarnings('ignore::RuntimeWarning')
   @pytest.mark.parametrize("dtype1", [
      "int64", 
      "float64", 
      "complex128",
   ])
   @pytest.mark.parametrize("dtype2", [
      "int64", 
      "float64", 
      "complex128",
   ])
   def test_astype(self, dtype1, dtype2):

       def fun(x, dtype):
           return tn.astype(x, dtype)

       x = data.tensor_dat(data.randn)(
              self.backend, "ijk", (2,3,4), dtype=dtype1
           )
       assert_grad(fun, submode="type")(x.tensor, dtype2) 


   @pytest.mark.parametrize("indnames, shape, minval, maxval", [
      ["ijk", (2,3,4), 0, 1], 
   ])
   def test_clip(self, indnames, shape, minval, maxval):

       def fun(x, minval, maxval):
           return tn.clip(x, minval, maxval)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, minval, maxval)


   @pytest.mark.parametrize("indnames, shape, inds", [
      ["ijk", (2,3,4), None], 
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   def test_flip(self, indnames, shape, inds):

       def fun(x, inds):
           return tn.flip(x, inds)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, inds)


   @pytest.mark.parametrize("indnames, shape, ind", [
      ["ijk", (2,3,4), None],
      ["ijk", (2,3,4), "j"],
   ])
   def test_cumsum(self, indnames, shape, ind):

       def fun(x, ind):
           return tn.cumsum(x, ind)

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       assert_grad(fun)(x.tensor, ind)


   @pytest.mark.parametrize("indnames, shape, positions", [
      ["ijk", (2,3,4), [(1,0,2), (0,2,1), (1,0,3)]],
   ])
   def test_getitem(self, indnames, shape, positions):

       def fun(x, pos):
           return x[pos]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )

       """
       pos = (1,0,2)

       pos = (
              ((1,0,1),), 
              ((0,2,0),),
              ((2,1,3),),
             )
       """

       """
       pos = (
              ((1,),), 
              ((0,),),
              ((2,),),
             )
       """

       for pos in positions:
           assert_grad(fun)(x.tensor, pos)


   @pytest.mark.parametrize("graddat", [
      data.sparsegrad_dat_001,
   ])
   @pytest.mark.parametrize("dtype", [
      "complex128",
   ])
   def test_ungetitem(self, graddat, dtype):

       def fun(x, pos, space):
           return tn.ungetitem(x, pos, space)

       w = graddat(
              self.backend, dtype, seed=1
           )   

       for i in range(len(w.pos)):

           x   = tn.astensor(w.vals[i])
           pos = w.pos[i]

           assert_grad(fun)(x, pos, tn.space(w.tensor))




###############################################################################
###                                                                         ###
###  Binary elementwise grads                                               ###
###                                                                         ###
###############################################################################


# --- Binary elementwise grads ---------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape", [
      ["ijk", (2,3,4)],
   ])
   @pytest.mark.parametrize("op", [
      "add", 
      "sub", 
      "mul", 
      "div", 
      "pow",
      "addgrads",
   ])
   def test_math(self, indnames, shape, op):

       fun = {
              "add":      lambda x, y: x + y,
              "sub":      lambda x, y: x - y,
              "mul":      lambda x, y: x * y,
              "div":      lambda x, y: x / y,
              "pow":      lambda x, y: x ** y,
              "addgrads": lambda x, y: tn.addgrads(x, y),
             }[op]

       x = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=1
           )
       y = data.tensor_dat(data.randn)(
              self.backend, indnames, shape, seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       assert_grad(fun, 0)(xtensor, ytensor)
       assert_grad(fun, 1)(xtensor, ytensor)


   @pytest.mark.parametrize("sampledat", [
      ardata.randuniform_real_dat_001,
   ])
   @pytest.mark.parametrize("op", [
      "mod", 
   ])
   def test_math_int(self, sampledat, op):

       fun = {
              "mod": lambda x, y: x % y,
             }[op]

       x = data.tensor_sample_dat(sampledat)(
              self.backend, boundaries=(1,11), seed=1
           )
       y = data.tensor_sample_dat(sampledat)(
              self.backend, boundaries=(1,11), seed=2
           )

       xtensor = x.tensor
       ytensor = tn.TensorGen(y.array, x.inds)

       assert_grad(fun, 0)(xtensor, ytensor)
       assert_grad(fun, 1)(xtensor, ytensor)




###############################################################################
###                                                                         ###
###  Ternary elementwise grads                                              ###
###                                                                         ###
###############################################################################


# --- Ternary elementwise grads --------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsElemwiseTernary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("indnames, shape, nvals", [
      ["ijk", (2,3,4), 5],
   ])
   def test_where(self, indnames, shape, nvals):

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

       def fun(w,x,y):
           return tn.where(w,x,y)

       assert_grad(fun, 0, submode="null")(wtensor, xtensor, ytensor)
       assert_grad(fun, 1                )(wtensor, xtensor, ytensor)
       assert_grad(fun, 2                )(wtensor, xtensor, ytensor)




###############################################################################
###                                                                         ###
###  Tensor reduction grads                                                 ###
###                                                                         ###
###############################################################################


# --- Tensor reduction grads  ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsReduction:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("xinds, xshape, inds", [
      ["ijk", (2,3,4), ""],
      ["ijk", (2,3,4), "i"],
      ["ijk", (2,3,4), "ki"],
      ["ijk", (2,3,4), "kij"],
   ])
   @pytest.mark.parametrize("op", [
      "amax", 
      "amin",
      "sumover",
   ])
   def test_reduce(self, xinds, xshape, inds, op):

       fun = {
              "amax":    lambda x, inds: tn.amax(x, inds=inds),
              "amin":    lambda x, inds: tn.amin(x, inds=inds),
              "sumover": lambda x, inds: tn.sumover(x, inds=inds),
             }[op]

       x = data.tensor_dat(data.randn)(
              self.backend, xinds, xshape
           )

       assert_grad(fun)(x.tensor, x.inds.map(*inds))




###############################################################################
###                                                                         ###
###  Tensor reindexing grads                                                ###
###                                                                         ###
###############################################################################


# --- Tensor reindexing grads  ---------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsReindexing:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   def test_reindex(self):

       def fun(x, indmap):
           return tn.reindex(x, indmap) 

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       p = IndexGen("p",5)

       a = IndexGen("a",4)
       b = IndexGen("b",2)
       c = IndexGen("c",5)  

       shape  = (2,3,4)
       inds   = (i,j,k)
       indmap = {k: a, i: b, p: c}

       w = data.array_dat(data.randn)(
              self.backend, shape
           )
       x = tn.TensorGen(w.array, inds)

       assert_grad(fun)(x, indmap)


   def test_reindex_001(self):

       def fun(x, indmap):
           return tn.reindex(x, indmap) 

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       p = IndexGen("p",5)

       a = IndexGen("a",4)
       b = IndexGen("b",2)
       c = IndexGen("c",5)  

       shape  = (2,3,4)
       inds   = (i,j,k) 
       indmap = {p: c}

       w = data.array_dat(data.randn)(
              self.backend, shape
           )
       x = tn.TensorGen(w.array, inds)

       assert_grad(fun)(x, indmap)


   @pytest.mark.parametrize("inds, shape, outinds", [
      ["ijkl", (2,3,4,5), "kjil"],
      ["ijkl", (2,3,4,5), "jlik"],
      ["ijkl", (2,3,4,5), "ijkl"],
   ])
   def test_transpose(self, inds, shape, outinds):

       def fun(x, *outinds):
           return tn.transpose(x, *outinds) 

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       assert_grad(fun)(x.tensor, *outinds)        


   @pytest.mark.parametrize("inds, shape, outinds", [
      ["ijkl", (2,3,4,5), "kjil"],
      ["ijkl", (2,3,4,5), "jlik"],
      ["ijkl", (2,3,4,5), "ijkl"],
   ])
   def test_htranspose(self, inds, shape, outinds):

       def fun(x, *outinds):
           return tn.htranspose(x, *outinds) 

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       assert_grad(fun, submode="real")(x.tensor, *outinds) 


   def test_fuse(self):

       def fun(x, indmap):
           return tn.fuse(x, indmap) 

       x = data.tensor_dat(data.randn)(
              self.backend, "ijkl", (2,3,4,5)
           )

       a = IndexGen("a",6)
       b = IndexGen("b",20)

       indmap = {
                 ("i","j"): a, 
                 ("k","l"): b,
                }

       assert_grad(fun)(x.tensor, indmap) 


   def test_fuse_001(self):

       def fun(x, indmap):
           return tn.fuse(x, indmap) 

       x = data.tensor_dat(data.randn)(
              self.backend, "ijklmn", (2,3,4,5,6,7)
           )

       a  = IndexGen("a",21)
       b  = IndexGen("b",40)
       m, = x.inds.map("m")

       indmap = {
                 ("j","n"): a, 
                 ("i","k","l"): b,
                }

       assert_grad(fun)(x.tensor, indmap) 


   def test_fuse_002(self):

       def fun(x, indmap):
           return tn.fuse(x, indmap) 

       x = data.tensor_dat(data.randn)(
              self.backend, "ijkl", (2,3,4,5)
           )

       a = IndexGen("a",120)

       indmap = {
                 ("k","i","j","l"): a,
                }

       assert_grad(fun)(x.tensor, indmap) 


   def test_split(self):

       def fun(x, indmap):
           return tn.split(x, indmap) 

       x = data.tensor_dat(data.randn)(
              self.backend, "ijkl", (4,3,24,5)
           )

       a = IndexGen("a",2)
       b = IndexGen("b",2)
       c = IndexGen("c",4)
       d = IndexGen("d",2)
       e = IndexGen("e",3)

       indmap = {
                 "i": (a, b),
                 "k": (c, d, e), 
                }

       assert_grad(fun)(x.tensor, indmap) 


   def test_split_001(self):

       def fun(x, indmap):
           return tn.split(x, indmap) 

       x = data.tensor_dat(data.randn)(
              self.backend, "a", (120,)
           )

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       l = IndexGen("l",5)

       indmap = {
                 "a": (k,i,j,l),
                }

       assert_grad(fun)(x.tensor, indmap) 


   @pytest.mark.parametrize("inds, shape, squeezed", [
      ["iajkbcl", (2,1,3,4,1,1,5), None],
      ["iajkbcl", (2,1,3,4,1,1,5), "ac"],
   ])
   def test_squeeze(self, inds, shape, squeezed):

       def fun(x, inds):
           return tn.squeeze(x, inds) 

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       assert_grad(fun)(x.tensor, squeezed) 


   @pytest.mark.parametrize("inds, shape, newinds", [
      ["ijkl", (2,3,4,5), "ab"],
      ["ijkl", (2,3,4,5), ""],
   ])
   def test_unsqueeze(self, inds, shape, newinds):

       def fun(x, inds):
           return tn.unsqueeze(x, inds) 

       w = data.indices_dat(
              newinds           + inds, 
              (1,)*len(newinds) + shape
           ) 
       v = data.array_dat(data.randn)(
              self.backend, shape
           ) 
       x = tn.TensorGen(v.array, w.inds.map(*inds))

       assert_grad(fun)(x, w.inds.map(*newinds))   


   @pytest.mark.parametrize("inds, shape, newinds, newsizes", [
      ["ijkl", (2,3,4,5), "ab", (6,7)],
      ["ijkl", (2,3,4,5), "", tuple()],
   ])
   def test_expand(self, inds, shape, newinds, newsizes):

       def fun(x, inds):
           return tn.expand(x, inds) 

       w = data.indices_dat(
              newinds  + inds, 
              newsizes + shape
           ) 
       v = data.array_dat(data.randn)(
              self.backend, shape
           ) 
       x = tn.TensorGen(v.array, w.inds.map(*inds))

       assert_grad(fun)(x, w.inds.map(*newinds)) 


   @pytest.mark.parametrize("inds, shape, ind, size", [
      ["ijkl",  (2,3,4,5),  "a",  120],
      ["i",     (3,),       "a",  3],
      ["i",     (1,),       "a",  1],
   ])
   def test_flatten(self, inds, shape, ind, size):

       def fun(x, ind):
           return tn.flatten(x, ind) 

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )
       ind = IndexGen(ind, size)

       assert_grad(fun)(x.tensor, ind) 




###############################################################################
###                                                                         ###
###  Tensor contraction grads                                               ###
###                                                                         ###
###############################################################################


# --- Tensor contraction grads  --------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsContraction:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),   (6,3,4)           ], ["ijk",  "kij",       ]],
      [[(3,4,6),   tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6),   (6,2,5)           ], ["ijk",  "klm",       ]],  
      [[(3,4,6),   (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]], 
      [[(3,4,5,6), (5,3)             ], ["ijkl", "ki",        ]],
   ])    
   def test_contract(self, shapes, inds):

       def fun(*xs):
           return tn.contract(*xs)

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert_grad(fun, i)(*w.tensors) 


   @pytest.mark.parametrize("shapes, inds, outinds", [
      [[(3,4,6),   (6,3,4)           ], ["ijk",  "kij",       ], ""    ], 
      [[(3,4,6),   (6,2,5)           ], ["ijk",  "klm",       ], "ijlm"],  
      [[(3,4,6),   (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"], "imq" ],
      [[(3,4,5,6), (5,3)             ], ["ijkl", "ki",        ], "jl"  ],
      [[(3,4,5,6), (5,3)             ], ["ijkl", "ki",        ], "jkl" ],
   ])    
   def test_contract_fixed(self, shapes, inds, outinds):

       def fun(*xs, product=None):
           return tn.contract(*xs, product=product)

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert_grad(fun, i)(*w.tensors, product=outinds) 


   @pytest.mark.parametrize("shape, inds, traceinds", [
      [(4,4),       "ij",    "ij" ], 
      [(3,3,3),     "ijk",   "ijk"], 
      [(3,4,4),     "ijk",   "jk" ], 
      [(3,4,5,6),   "ijkl",  "ik" ], 
      [(3,4,5,6),   "ijkl",  "ki" ], 
      [(3,4,5,6,7), "ijklm", "ikl"],
   ])
   def test_trace(self, shape, inds, traceinds):

       def fun(x, inds):
           return tn.trace(x, inds)

       x = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           )

       assert_grad(fun)(x.tensor, traceinds)  


   @pytest.mark.parametrize("shapes, inds, outshape, outinds", [ 
      [[(2,5,4), (3,6,7)], ["ijk", "lmn"], (6,30,28), "abc"],  
   ])   
   def test_kron(self, shapes, inds, outshape, outinds):

       def fun(x, y, indmap):
           return tn.kron(x, y, indmap)

       v = data.indices_dat(outinds, outshape)
       w = data.ntensor_dat(data.randn)(self.backend, inds, shapes)

       indmap = dict(zip(zip(*inds), v.inds))

       assert_grad(fun)(*w.tensors, indmap) 




