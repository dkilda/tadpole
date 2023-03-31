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

import tadpole.array.backends    as backends
import tadpole.tensor.reindexing as reidx
import tadpole.tensor.engine     as tne 

import tests.tensor.fakes as fake
import tests.tensor.data  as data


from tadpole.tensor.types import (
   Pluggable,
   Tensor, 
   Space,
)


from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)



###############################################################################
###                                                                         ###
###  Tensor reindexing engine and operator                                  ###
###                                                                         ###
###############################################################################


# --- Tensor reindexing operator -------------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTensorReindex:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Reindexing and reshaping methods --- #

   def test_reindex(self):

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       p = IndexGen("p",5)

       a = IndexGen("a",4)
       b = IndexGen("i",2)
       c = IndexGen("c",5)  

       shape = (2,3,4)
       inds1 = (i,j,k)
       inds2 = (b,j,a)   

       w = data.array_dat(data.randn)(
              self.backend, shape
           )

       x1 = tn.TensorGen(w.array, inds1)
       x2 = tn.TensorGen(w.array, inds2)

       indmap = {k: a, i: b, p: c}

       assert tn.reindex(x1, indmap) == x2


   def test_reindex_001(self):

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       p = IndexGen("p",5)

       a = IndexGen("a",4)
       b = IndexGen("i",2)
       c = IndexGen("c",5)  

       shape = (2,3,4)
       inds  = (i,j,k) 

       w = data.array_dat(data.randn)(
              self.backend, shape
           )
       x = tn.TensorGen(w.array, inds)

       assert tn.reindex(x, {p: c}) == x
       

   def test_reindex_fail(self):

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       p = IndexGen("p",5)

       a = IndexGen("a",6)
       b = IndexGen("i",2)
       c = IndexGen("c",5)  

       shape = (2,3,4)
       inds1 = (i,j,k)
       inds2 = (b,j,a)   

       w = data.array_dat(data.randn)(
              self.backend, shape
           )
       x = tn.TensorGen(w.array, inds1)

       try:
           out = tn.reindex(x, {k: a, i: b, p: c})
       except ValueError:
           assert True
       else:
           assert False


   @pytest.mark.parametrize("inds, shape, outinds, outaxes", [
      ["ijkl", (2,3,4,5), "kjil", (2,1,0,3)],
      ["ijkl", (2,3,4,5), "jlik", (1,3,0,2)],
      ["ijkl", (2,3,4,5), "ijkl", (0,1,2,3)],
   ])
   def test_transpose(self, inds, shape, outinds, outaxes):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       out = tn.transpose(w.tensor, *outinds)
       ans = ar.transpose(w.array, outaxes) 
       ans = tn.TensorGen(ans, w.inds.map(*outinds))

       assert out == ans


   @pytest.mark.parametrize("inds, shape, outinds, outaxes", [
      ["ijkl", (2,3,4,5), "kjil", (2,1,0,3)],
      ["ijkl", (2,3,4,5), "jlik", (1,3,0,2)],
      ["ijkl", (2,3,4,5), "ijkl", (0,1,2,3)],
   ])
   def test_htranspose(self, inds, shape, outinds, outaxes):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       out = tn.htranspose(w.tensor, *outinds)
       ans = ar.transpose(ar.conj(w.array), outaxes) 
       ans = tn.TensorGen(ans, w.inds.map(*outinds))

       assert out == ans


   def test_fuse(self):

       inds  = "ijkl"
       shape = (2,3,4,5)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       a = IndexGen("a",6)
       b = IndexGen("b",20)

       indmap = {
                 ("i","j"): a, 
                 ("k","l"): b,
                }

       out = tn.fuse(w.tensor, indmap)
       ans = ar.reshape(ar.transpose(w.array, (0,1,2,3)), (6,20))
       ans = tn.TensorGen(ans, (a,b))

       assert out == ans


   def test_fuse_001(self):

       inds  = "ijklmn"
       shape = (2,3,4,5,6,7)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       a = IndexGen("a",21)
       b = IndexGen("b",40)
       m = w.inds.map("m")[0]

       indmap = {
                 ("j","n"): a, 
                 ("i","k","l"): b,
                }

       out = tn.fuse(w.tensor, indmap)
       ans = ar.reshape(ar.transpose(w.array, (1,5,0,2,3,4)), (21,40,6))
       ans = tn.TensorGen(ans, (a,b,m))

       assert out == ans


   def test_fuse_002(self):

       inds  = "ijkl"
       shape = (2,3,4,5)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       a = IndexGen("a",120)

       indmap = {
                 ("k","i","j","l"): a,
                }

       out = tn.fuse(w.tensor, indmap)
       ans = ar.reshape(ar.transpose(w.array, (2,0,1,3)), (120,))
       ans = tn.TensorGen(ans, (a,))

       assert out == ans


   def test_split(self):

       inds  = "ijkl"
       shape = (4,3,24,5)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       i,j,k,l = w.inds.map(*inds)

       a = IndexGen("a",2)
       b = IndexGen("b",2)
       c = IndexGen("c",4)
       d = IndexGen("d",2)
       e = IndexGen("e",3)

       indmap = {
                 "i": (a, b),
                 "k": (c, d, e), 
                }

       out = tn.split(w.tensor, indmap)
       ans = ar.reshape(w.array, (2,2,3,4,2,3,5))
       ans = tn.TensorGen(ans, (a,b,j,c,d,e,l))

       assert out == ans


   def test_split_001(self):

       inds  = "a"
       shape = (120,)

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       a = w.inds.map(*inds)

       i = IndexGen("i",2)
       j = IndexGen("j",3)
       k = IndexGen("k",4)
       l = IndexGen("l",5)

       indmap = {
                 "a": (k,i,j,l),
                }

       out = tn.split(w.tensor, indmap)
       ans = ar.reshape(w.array, (4,2,3,5))
       ans = tn.TensorGen(ans, (k,i,j,l))

       assert out == ans


   @pytest.mark.parametrize("inds, shape, squeezed, axes, output", [
      ["iajkbcl", (2,1,3,4,1,1,5), None, None,  "ijkl"],
      ["iajkbcl", (2,1,3,4,1,1,5), "ac", (1,5), "ijkbl"],
   ])
   def test_squeeze(self, inds, shape, squeezed, axes, output):

       w = data.tensor_dat(data.randn)(
              self.backend, inds, shape
           ) 

       if   squeezed is None:

            out = tn.squeeze(w.tensor)
            ans = ar.squeeze(w.array) 
            ans = tn.TensorGen(ans, w.inds.map(*output))

       else:
            out = tn.squeeze(w.tensor, squeezed)
            ans = ar.squeeze(w.array, axes) 
            ans = tn.TensorGen(ans, w.inds.map(*output))

       assert out == ans


   @pytest.mark.parametrize("inds, shape, newinds", [
      ["ijkl", (2,3,4,5), "ab"],
      ["ijkl", (2,3,4,5), ""],
   ])
   def test_unsqueeze(self, inds, shape, newinds): 

       inds1   = newinds + inds
       shape1  = (1,)*len(newinds) + shape
       newaxes = tuple(range(len(newinds))) 

       v = data.array_dat(data.randn)(
              self.backend, shape
           ) 
       w = data.tensor_dat(data.randn)(
              self.backend, inds1, shape1
           ) 
       x = tn.TensorGen(v.array, w.inds.map(*inds))

       out = tn.unsqueeze(x,       w.inds.map(*newinds))
       ans = ar.unsqueeze(v.array, newaxes) 
       ans = tn.TensorGen(ans,     w.inds.map(*inds1))

       assert out == ans


   @pytest.mark.parametrize("inds, shape, newinds, newsizes", [
      ["ijkl", (2,3,4,5), "ab", (6,7)],
      ["ijkl", (2,3,4,5), "", tuple()],
   ])
   def test_expand(self, inds, shape, newinds, newsizes): 

       inds1   = newinds  + inds
       shape1  = newsizes + shape
       newaxes = tuple(range(len(newinds))) 

       v = data.array_dat(data.randn)(
              self.backend, shape
           ) 
       w = data.tensor_dat(data.randn)(
              self.backend, inds1, shape1
           ) 
       x = tn.TensorGen(v.array, w.inds.map(*inds))

       out = tn.expand(x, w.inds.map(*newinds))
       ans = ar.broadcast_to(v.array, shape1) 
       ans = tn.TensorGen(ans, w.inds.map(*inds1))

       assert out == ans




