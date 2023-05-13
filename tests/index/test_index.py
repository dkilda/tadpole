#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np
import copy  as cp

import tadpole.util  as util
import tadpole.index as tid

import tests.index.fakes as fake
import tests.index.data  as data


from tadpole.index import (
   Index,
   IndexGen,  
   IndexLit,
   IndexOne,
   Indices,
)




###############################################################################
###                                                                         ###
###  Tensor index                                                           ###
###                                                                         ###
###############################################################################


# --- General Index --------------------------------------------------------- #

class TestIndexGen:

   # --- Copying (forbidden to enforce uniqueness) --- #

   @pytest.mark.parametrize("tags, size", [
      ["i", 2],
   ])
   def test_copy(self, tags, size):

       i = IndexGen(tags, size)

       assert cp.copy(i)     is i
       assert cp.deepcopy(i) is i


   # --- Equality, hashing, size --- #

   @pytest.mark.parametrize("tags, size", [
      ["i", 2],
   ])
   def test_eq(self, tags, size):

       i = IndexGen(tags, size)
       j = IndexGen(tags, size)

       assert i != j


   @pytest.mark.parametrize("tags, size", [
      ["i", 2],
   ])
   def test_hash(self, tags, size):

       i = IndexGen(tags, size)
       j = IndexGen(tags, size)

       assert hash(i) != hash(j)


   @pytest.mark.parametrize("tags, size, output", [
      ["i", None, 1],
      ["i", 1,    1],
      ["i", 5,    5],
   ])
   def test_len(self, tags, size, output):

       if  size is None:
           i = IndexGen(tags) 
       else:
           i = IndexGen(tags, size)

       assert len(i) == output
       

   # --- General methods --- #

   @pytest.mark.parametrize("tags, inp, out", [
      ["i",        ["i"],      True],
      [("i", "j"), ["i"],      True],
      [("i", "j"), ["j"],      True],
      [("i", "j"), ["i", "j"], True],
      [("i", "j"), ["j", "i"], True],
      [("i", "j"), ["k"],      False],
      [("i", "j"), ["k", "l"], False],
      [("i", "j"), ["i", "k"], False],
      [("i", "j"), ["j", "k"], False],
      [("i", "j"), ["k", "i"], False],
      [("i", "j"), ["k", "j"], False],
   ])
   @pytest.mark.parametrize("size", [5])
   def test_all(self, tags, size, inp, out):

       i = IndexGen(tags, size)

       assert i.all(*inp) == out


   @pytest.mark.parametrize("tags, inp, out", [
      ["i",        ["i"],      True],
      [("i", "j"), ["i"],      True],
      [("i", "j"), ["j"],      True],
      [("i", "j"), ["i", "j"], True],
      [("i", "j"), ["j", "i"], True],
      [("i", "j"), ["k"],      False],
      [("i", "j"), ["k", "l"], False],
      [("i", "j"), ["i", "k"], True],
      [("i", "j"), ["j", "k"], True],
      [("i", "j"), ["k", "i"], True],
      [("i", "j"), ["k", "j"], True],
   ])
   @pytest.mark.parametrize("size", [5])
   def test_any(self, tags, size, inp, out):

       i = IndexGen(tags, size)

       assert i.any(*inp) == out


   @pytest.mark.parametrize("size, start, end", [
      [5, 0, 5],
      [5, 1, 5],
      [5, 2, 7],
      [1, 0, 5],
      [5, 0, 1],
   ])
   @pytest.mark.parametrize("tags", [
      "i", 
      ("i", "j"),
   ])
   def test_resized(self, tags, size, start, end):

       i  = IndexGen(tags, size)
       i1 = i.resized(start, end)

       assert i1 != i
       assert len(i1) == end - start
       assert i1.all(*tags)

       assert len(i) == size
       assert i.all(*tags)


   @pytest.mark.parametrize("size", [5])
   @pytest.mark.parametrize("tags, tags1", [
      ["i",        "i"],
      ["i",        "j"], 
      ["i",        ("i", "j")], 
      [("i", "j"), "k"],
      [("i", "j"), ("i", "k")],
      [("i", "j"), ("m", "k")],
      [("i", "j"), ("i", "j")],
   ])
   def test_retagged(self, size, tags, tags1):

       i  = IndexGen(tags, size)
       i1 = i.retagged(tags1)

       assert i1 != i
       assert len(i1) == size
       assert i1.all(*tags1)

       assert len(i) == size
       assert i.all(*tags)




# --- Literal Index --------------------------------------------------------- #

class TestIndexLit:

   # --- Copying (forbidden to enforce uniqueness) --- #

   @pytest.mark.parametrize("tags, size", [
      ["i", 2],
   ])
   def test_copy(self, tags, size):

       i = IndexLit(tags, size)

       assert cp.copy(i)     is i
       assert cp.deepcopy(i) is i


   # --- Equality, hashing, size --- #

   @pytest.mark.parametrize("tags1, tags2, size, res", [
      ["i", "i", 2, True],
      ["i", "j", 2, False],
   ])
   def test_eq(self, tags1, tags2, size, res):

       i = IndexLit(tags1, size)
       j = IndexLit(tags2, size)

       if res:
          assert i == j
       else:
          assert i != j


   @pytest.mark.parametrize("tags1, tags2, size, res", [
      ["i", "i", 2, True],
      ["i", "j", 2, False],
   ])
   def test_hash(self, tags1, tags2, size, res):

       i = IndexLit(tags1, size)
       j = IndexLit(tags2, size)

       if res:
          assert hash(i) == hash(j)
       else:
          assert hash(i) != hash(j)


   @pytest.mark.parametrize("tags, size, output", [
      ["i", None, 1],
      ["i", 1,    1],
      ["i", 5,    5],
   ])
   def test_len(self, tags, size, output):

       if  size is None:
           i = IndexLit(tags) 
       else:
           i = IndexLit(tags, size)

       assert len(i) == output
       

   # --- General methods --- #

   @pytest.mark.parametrize("tags, inp, out", [
      ["i",        ["i"],      True],
      [("i", "j"), ["i"],      True],
      [("i", "j"), ["j"],      True],
      [("i", "j"), ["i", "j"], True],
      [("i", "j"), ["j", "i"], True],
      [("i", "j"), ["k"],      False],
      [("i", "j"), ["k", "l"], False],
      [("i", "j"), ["i", "k"], False],
      [("i", "j"), ["j", "k"], False],
      [("i", "j"), ["k", "i"], False],
      [("i", "j"), ["k", "j"], False],
   ])
   @pytest.mark.parametrize("size", [5])
   def test_all(self, tags, size, inp, out):

       i = IndexLit(tags, size)

       assert i.all(*inp) == out


   @pytest.mark.parametrize("tags, inp, out", [
      ["i",        ["i"],      True],
      [("i", "j"), ["i"],      True],
      [("i", "j"), ["j"],      True],
      [("i", "j"), ["i", "j"], True],
      [("i", "j"), ["j", "i"], True],
      [("i", "j"), ["k"],      False],
      [("i", "j"), ["k", "l"], False],
      [("i", "j"), ["i", "k"], True],
      [("i", "j"), ["j", "k"], True],
      [("i", "j"), ["k", "i"], True],
      [("i", "j"), ["k", "j"], True],
   ])
   @pytest.mark.parametrize("size", [5])
   def test_any(self, tags, size, inp, out):

       i = IndexLit(tags, size)

       assert i.any(*inp) == out


   @pytest.mark.parametrize("size, start, end", [
      [5, 0, 5],
      [5, 1, 5],
      [5, 2, 7],
      [1, 0, 5],
      [5, 0, 1],
   ])
   @pytest.mark.parametrize("tags", [
      "i", 
      ("i", "j"),
   ])
   def test_resized(self, tags, size, start, end):

       i  = IndexLit(tags, size)
       i1 = i.resized(start, end)

       assert i1 != i
       assert len(i1) == end - start
       assert i1.all(*tags)

       assert len(i) == size
       assert i.all(*tags)


   @pytest.mark.parametrize("size", [5])
   @pytest.mark.parametrize("tags, tags1", [
      ["i",        "i"],
      ["i",        "j"], 
      ["i",        ("i", "j")], 
      [("i", "j"), "k"],
      [("i", "j"), ("i", "k")],
      [("i", "j"), ("m", "k")],
      [("i", "j"), ("i", "j")],
   ])
   def test_retagged(self, size, tags, tags1):

       i  = IndexLit(tags, size)
       i1 = i.retagged(tags1)

       if tags == tags1:
          assert i1 == i
       else:
          assert i1 != i

       assert len(i1) == size
       assert i1.all(*tags1)

       assert len(i) == size
       assert i.all(*tags)




# --- Singleton Index ------------------------------------------------------- #

class TestIndexOne:

   # --- Copying (forbidden to enforce uniqueness) --- #

   def test_copy(self):

       i = IndexOne()

       assert cp.copy(i)     is i
       assert cp.deepcopy(i) is i


   # --- Equality, hashing, size --- #

   def test_eq(self):

       i = IndexOne()
       j = IndexOne()

       assert i == j


   def test_hash(self):

       i = IndexOne()
       j = IndexOne()

       assert hash(i) == hash(j)


   def test_len(self):

       i = IndexOne() 

       assert len(i) == 1
       

   # --- General methods --- #

   @pytest.mark.parametrize("inp", [
      ["i"],         
      ["i", "j"],
   ])
   def test_all(self, inp):

       i = IndexOne()

       assert i.all(*inp) == False


   @pytest.mark.parametrize("inp", [
      ["i"],         
      ["i", "j"],
   ])
   def test_any(self, inp):

       i = IndexOne()

       assert i.any(*inp) == False


   @pytest.mark.parametrize("start, end", [
      [0, 5],
      [1, 5],
      [2, 7],
      [0, 5],
      [0, 1],
   ])
   def test_resized(self, start, end):

       i  = IndexOne()
       i1 = i.resized(start, end)

       assert i1 != i
       assert len(i1) == end - start


   @pytest.mark.parametrize("tags", [
      "i",
      ("i", "j"), 
   ])
   def test_retagged(self, tags):

       i  = IndexOne()
       i1 = i.retagged(tags)

       assert i1 != i
       assert len(i1) == 1
       assert i1.all(*tags)




###############################################################################
###                                                                         ###
###  Collection of tensor indices with extra functionality                  ###
###  (operations acting on groups of Index objects).                        ###
###                                                                         ###
###############################################################################


# --- Indices --------------------------------------------------------------- #

class TestIndices:

   # --- Copying --- #

   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_copy(self, tags, shape):

       w   = data.indices_dat(tags, shape)
       out = w.inds.copy()

       assert out == w.inds
       assert out is not w.inds

       
   # --- Container methods --- #

   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_eq(self, tags, shape):

       x = data.indices_dat(tags, shape)
       y = data.indices_dat(tags, shape)

       assert x.inds != y.inds


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_hash(self, tags, shape):

       x = data.indices_dat(tags, shape)
       y = data.indices_dat(tags, shape)

       assert hash(x.inds) != hash(y.inds)


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_len(self, tags, shape):

       x = data.indices_dat(tags, shape)

       assert len(x.inds) == len(shape)


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_contains(self, tags, shape):

       x = data.indices_dat(tags, shape)

       for i in x.inds:
           assert i in x.inds 

       j = IndexGen(tags[0], shape[0])
       assert j not in x.inds


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_iter(self, tags, shape):

       x   = data.indices_dat(tags, shape)
       xit = iter(x.inds)

       assert tuple(xit) == tuple(x.indlist)


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_reversed(self, tags, shape):

       x    = data.indices_dat(tags, shape)
       xrit = reversed(x.inds)

       assert tuple(xrit) == tuple(reversed(x.indlist))


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_getitem(self, tags, shape):

       x = data.indices_dat(tags, shape)

       for pos in range(len(shape)):
           assert x.inds[pos] == x.indlist[pos]
       

   # --- Properties --- #

   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_size(self, tags, shape):

       w = data.indices_dat(tags, shape)

       assert w.inds.size == np.prod(shape)


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_ndim(self, tags, shape):

       w = data.indices_dat(tags, shape)

       assert w.inds.ndim == len(shape)


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_shape(self, tags, shape):

       w = data.indices_dat(tags, shape)

       assert w.inds.shape == shape


   # --- Index container behavior --- #

   @pytest.mark.parametrize("tags, shape, select, result", [ 
      ["ijk", (2,3,4), ["i"     ], [0]],
      ["ijk", (2,3,4), ["i", "j"], [ ]],
      ["ijk", (2,3,4), ["k", "i"], [ ]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1","i2"], [0   ]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1"     ], [0, 2]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["k2"     ], [2   ]],
   ])      
   def test_all(self, tags, shape, select, result):

       w = data.indices_dat(tags, shape)
       
       out = w.inds.all(*select)
       ans = tuple(w.indlist[res] for res in result)

       assert out == ans 


   @pytest.mark.parametrize("tags, shape, select, result", [ 
      ["ijk", (2,3,4), ["i"],      [0   ]],
      ["ijk", (2,3,4), ["i", "j"], [0, 1]],
      ["ijk", (2,3,4), ["k", "i"], [0, 2]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1","i2"], [0, 2]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1"     ], [0, 2]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["k2"     ], [2   ]],
   ])      
   def test_any(self, tags, shape, select, result):

       w = data.indices_dat(tags, shape)
       
       out = w.inds.any(*select)
       ans = tuple(w.indlist[res] for res in result)

       assert out == ans 


   @pytest.mark.parametrize("tags, shape, select, result", [ 
      ["ijk", (2,3,4), "i",   [0    ]],
      ["ijk", (2,3,4), "ik",  [0,2  ]],
      ["ijk", (2,3,4), "ijk", [0,1,2]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), [("i1","i2")], [0]],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1"       ], [ ]],
   ])
   def test_map(self, tags, shape, select, result):

       w = data.indices_dat(tags, shape)
       
       out = w.inds.map(*select)
       ans = tuple(w.indlist[res] for res in result)

       assert out == ans 


   @pytest.mark.parametrize("tags, shape, select, result", [ 
      ["ijk", (2,3,4), [0],     (0,   )],
      ["ijk", (2,3,4), [0,2],   (0,2  )],
      ["ijk", (2,3,4), [0,1,2], (0,1,2)],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), [0], (0, )],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), [], tuple()],
   ])
   def test_axes(self, tags, shape, select, result):

       w = data.indices_dat(tags, shape)

       select = [w.indlist[sel] for sel in select]
       
       assert w.inds.axes(*select) == result


   @pytest.mark.parametrize("tags, shape, select, result", [ 
      ["ijk", (2,3,4), "i",   (0,   )],
      ["ijk", (2,3,4), "ik",  (0,2  )],
      ["ijk", (2,3,4), "ijk", (0,1,2)],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), [("i1","i2")], (0, )],
      [[("i1","i2"), "j", ("i1", "k2")], (2,3,4), ["i1"       ], tuple()],
   ])
   def test_axes_001(self, tags, shape, select, result):

       w = data.indices_dat(tags, shape)
       
       assert w.inds.axes(*select) == result


   # --- Out-of-place modifications --- #

   @pytest.mark.parametrize("tags, shape, remove, result", [ 
      ["ijk", (2,3,4), [0],     [1,2]],
      ["ijk", (2,3,4), [0,2],   [1  ]],
      ["ijk", (2,3,4), [0,1,2], [   ]],
   ])
   def test_remove(self, tags, shape, remove, result):

       w = data.indices_dat(tags, shape)

       remove = (w.indlist[rem] for rem in remove)
       result = (w.indlist[res] for res in result)

       out = w.inds.remove(*remove)
       ans = Indices(*result)   

       assert out == ans


   @pytest.mark.parametrize("tags, shape, select, add, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3], [4,3,0    ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [4, ], [4,0,1,2  ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [3,4], [3,4,0,1,2]],
   ])
   def test_add(self, tags, shape, select, add, result):

       w = data.indices_dat(tags, shape)

       select = (w.indlist[s] for s in select)       
       add    = (w.indlist[a] for a in add)
       result = (w.indlist[r] for r in result)

       x = Indices(*select)

       assert x.add(*add) == Indices(*result) 


   @pytest.mark.parametrize("tags, shape, select, add, axis, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3], 1, [0,4,3    ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [4, ], 1, [0,4,1,2  ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [3,4], 1, [0,3,4,1,2]],
   ])
   def test_add_axis(self, tags, shape, select, add, axis, result):

       w = data.indices_dat(tags, shape)

       select = (w.indlist[s] for s in select)       
       add    = (w.indlist[a] for a in add)
       result = (w.indlist[r] for r in result)

       x = Indices(*select)

       assert x.add(*add, axis=axis) == Indices(*result) 


   @pytest.mark.parametrize("tags, shape, select, push, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3], [0,4,3    ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [4, ], [0,1,2,4  ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [3,4], [0,1,2,3,4]],
   ])
   def test_push(self, tags, shape, select, push, result):

       w = data.indices_dat(tags, shape)

       select = (w.indlist[s] for s in select)       
       push   = (w.indlist[p] for p in push)
       result = (w.indlist[r] for r in result)

       x = Indices(*select)

       assert x.push(*push) == Indices(*result) 


   # --- Set arithmetic --- #

   @pytest.mark.parametrize("tags, shape, inds1, inds2, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3],     []],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,2],     [0,2]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,1,2,3], [0,1,2]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,1,2],   [0,1,2]],
   ])
   def test_and(self, tags, shape, inds1, inds2, result):

       w = data.indices_dat(tags, shape)

       inds1  = Indices(*(w.indlist[i] for i in inds1))      
       inds2  = Indices(*(w.indlist[i] for i in inds2))
       result = Indices(*(w.indlist[r] for r in result))

       assert (inds1 & inds2) == result


   @pytest.mark.parametrize("tags, shape, inds1, inds2, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3],  [0,4,3]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,2],  [0,1,2,0,2]],
   ])
   def test_or(self, tags, shape, inds1, inds2, result):

       w = data.indices_dat(tags, shape)

       inds1  = Indices(*(w.indlist[i] for i in inds1))      
       inds2  = Indices(*(w.indlist[i] for i in inds2))
       result = Indices(*(w.indlist[r] for r in result))

       assert (inds1 | inds2) == result


   @pytest.mark.parametrize("tags, shape, inds1, inds2, result", [ 
      ["ijklm", (2,3,4,5,6), [0],     [4,3],     [0]],
      ["ijklm", (2,3,4,5,6), [4,3],   [0],       [4,3]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,2],     [1]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,1,2,3], [ ]],
      ["ijklm", (2,3,4,5,6), [0,1,2], [0,1,2],   [ ]],
   ])
   def test_xor(self, tags, shape, inds1, inds2, result):

       w = data.indices_dat(tags, shape)

       inds1  = Indices(*(w.indlist[i] for i in inds1))      
       inds2  = Indices(*(w.indlist[i] for i in inds2))
       result = Indices(*(w.indlist[r] for r in result))

       assert (inds1 ^ inds2) == result




###############################################################################
###                                                                         ###
###  Index operations                                                       ###
###                                                                         ###
###############################################################################


# --- Basic index info ------------------------------------------------------ #

class TestIndexOperations:

   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_shapeof(self, tags, shape):

       w = data.indices_dat(tags, shape)

       assert tid.shapeof(*w.indlist) == shape


   @pytest.mark.parametrize("tags, shape", [ 
      ["ijk", (2,3,4)],
   ])      
   def test_sizeof(self, tags, shape):

       w = data.indices_dat(tags, shape)

       assert tid.sizeof(*w.indlist) == np.prod(shape)




