#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
import itertools
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn
import tadpole.index    as tid

import tadpole.tensorwrap.container as tc

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

from tadpole.tensorwrap.container import (
   NullGrad,
   SparseGrad,
   ContainerGen,
   ContainerSpace,
)




###############################################################################
###                                                                         ###
###  Container space                                                        ###
###                                                                         ###
###############################################################################


# --- ContainerSpace -------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestContainerSpace:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Fill space with data --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_fillwith(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       out = w.space.fillwith(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   # --- Reshape space --- #

   @pytest.mark.parametrize("shapes1, inds1, shapes2, inds2", [
      [[(3,4,6),                 ], ["ijk",            ], [(5,2,3),                 ], ["abc"             ]],
      [[(3,4,6), (6,2,5)         ], ["ijk",       "klm"], [(5,2,3),          (4,2,7)], ["abc",       "dbe"]], 
      [[(3,4,6), tuple(), (6,2,5)], ["ijk", "",   "klm"], [(5,2,3), (3,2),   (4,2,7)], ["abc", "cb", "dbe"]], 
      [[(3,4,6), (3,2),   (6,2,5)], ["ijk", "il", "klm"], [(5,2,3), tuple(), (4,2,7)], ["abc", "",   "dbe"]], 
   ]) 
   def test_reshape(self, shapes1, inds1, shapes2, inds2):

       w = data.container_dat(data.randn)(
              self.backend, inds1, shapes1, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds2, shapes2, seed=20
           )

       inds = [Indices(*v.inds.map(*i2)) for i2 in inds2]
       out  = w.space.reshape(inds)
       ans  = v.space

       assert out == ans


   # --- Gradient factories --- #

   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_sparsegrad(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.space.sparsegrad(w.pos, w.vals) == w.container


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_nullgrad(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.space.nullgrad() == w.container

       
   # --- Container factories --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_zeros(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.space.zeros()
       ans = ContainerGen([tspace.zeros() for tspace in w.tensorspaces])

       assert out == ans 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_ones(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.space.ones()
       ans = ContainerGen([tspace.ones() for tspace in w.tensorspaces])

       assert out == ans 


   @pytest.mark.parametrize("shapes, inds, positions", [
      [[(2,3),      ], ["ij",     ], [[(0,0)       ], [(0,2)       ], [(1,0)       ]]], 
      [[(2,3), (3,4)], ["ij", "jk"], [[(0,0), (1,2)], [(0,1), (0,2)], [(1,0), (1,0)]]], 
   ]) 
   def test_unit(self, shapes, inds, positions):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:

           out = w.space.unit(pos)
           ans = [tspace.unit(pos[i]) 
                  for i, tspace in enumerate(w.tensorspaces)]
           ans = ContainerGen(ans)

           assert out == ans 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_rand(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       seed = range(len(w.tensors))

       out = w.space.rand(seed=seed)
       ans = [tspace.rand(seed=seed[i]) 
              for i, tspace in enumerate(w.tensorspaces)]
       ans = ContainerGen(ans)

       assert out == ans 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_randn(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       seed = range(len(w.tensors))

       out = w.space.randn(seed=seed)
       ans = [tspace.randn(seed=seed[i]) 
              for i, tspace in enumerate(w.tensorspaces)]
       ans = ContainerGen(ans)

       assert out == ans 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_randuniform(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       seed       = range(len(w.tensors))
       boundaries = [(0,1)]*len(w.tensors) 

       out = w.space.randuniform(boundaries,    seed=seed)
       ans = [tspace.randuniform(boundaries[i], seed=seed[i]) 
              for i, tspace in enumerate(w.tensorspaces)]
       ans = ContainerGen(ans)

       assert out == ans 


   @pytest.mark.parametrize("shapes, inds", [
      [[(2,3),      ], ["ij",     ]], 
      [[(2,3), (3,4)], ["ij", "jk"]], 
   ]) 
   def test_units(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.space.units()
       ans = [tspace.units() for i, tspace in enumerate(w.tensorspaces)]
       ans = ContainerGen(ans)

       for outi, ansi in zip(out, ans):
           assert tuple(outi) == tuple(ansi)


   @pytest.mark.parametrize("shapes, inds", [
      [[(2,3),      ], ["ij",     ]], 
      [[(2,3), (3,4)], ["ij", "jk"]], 
   ]) 
   def test_basis(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.space.basis()
       ans = [tspace.basis() for i, tspace in enumerate(w.tensorspaces)]
       ans = ContainerGen(ans)

       for outi, ansi in zip(out, ans):
           assert tuple(outi) == tuple(ansi)


   # --- Space properties --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_dtype(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.space.dtype == tuple("complex128" for _ in shapes)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_size(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.space.size == tuple(np.prod(shape) for shape in shapes)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_ndim(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.space.ndim == tuple(len(shape) for shape in shapes)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_shape(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.space.shape == tuple(shapes)


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert len(w.space) == len(w.tensorspaces)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensorspaces)):
           assert w.tensorspaces[i] in w.space 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.space[i] is w.tensorspaces[i]


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert tuple(w.space) == tuple(w.tensorspaces)




###############################################################################
###                                                                         ###
###  Special container types for gradients                                  ###
###                                                                         ###
###############################################################################


# --- Null gradient --------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestNullGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.todense() == w.dense


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_tonull(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.tonull() == w.container


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_copy(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_withdata(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.null_container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_space(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_item(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container.item(i) == w.tensorspaces[i].zeros()


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.tensors[i] not in w.container 


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container[i] == w.tensorspaces[i].zeros() 


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert tuple(w.container) == tuple(w.dense)




# --- Sparse gradient ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestSparseGrad:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_todense(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.todense() == w.dense


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_tonull(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.tonull() == NullGrad(w.space)


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_copy(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_withdata(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_space(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_item(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.container.item(i) is w.tensors[i]
           else:
                assert w.container.item(i) == w.tensorspaces[i].zeros()


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_len(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_contains(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.tensors[i] in w.container 
           else:
                assert w.tensors[i] not in w.container 


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_getitem(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       for i in range(len(w.tensors)):
           if   i in pos:
                assert w.container[i] is w.tensors[i]
           else:
                assert w.container[i] == w.tensorspaces[i].zeros() 


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_iter(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       assert tuple(w.container) == tuple(w.dense)




###############################################################################
###                                                                         ###
###  General tensor container                                               ###
###                                                                         ###
###############################################################################


# --- General tensor container ---------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestContainerGen:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_todense(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.todense() == w.container


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_tonull(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.tonull() == NullGrad(w.space)


   # --- Container methods --- #

   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_copy(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )
 
       out = w.container.copy()
      
       assert out == w.container
       assert out is not w.container


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_withdata(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       out = w.container.withdata(v.arrays)
       ans = ContainerGen([
                tn.TensorGen(v.arrays[i], w.inds.map(*inds[i])) 
                for i in range(len(shapes))
             ])

       assert out == ans


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_space(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert w.container.space() == w.space


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_item(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container.item(i) is w.tensors[i]


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_len(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert len(w.container) == len(w.tensors)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_contains(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.tensors[i] in w.container 


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for i in range(len(w.tensors)):
           assert w.container[i] is w.tensors[i]


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem_001(self, shapes, inds):

       w = data.ntensor_dat(data.randn)(
              self.backend, inds, shapes
           )
       x = ContainerGen(w.tensors[:-1])

       for i in range(len(w.tensors) - 1):
           assert x[i] is w.tensors[i]
           assert not (x[i] == w.tensors[i+1])


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_iter(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       assert tuple(w.container) == tuple(w.tensors)




###############################################################################
###                                                                         ###
###  Gradient accumulation                                                  ###
###                                                                         ###
###############################################################################


# --- Gradient accumulation ------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradAccum:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",              ]], 
      [[(3,4,6), (6,2,5)           ], ["ijk", "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk", "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(x, y)
       ans = [tn.addgrads(xi, yi) for xi, yi in zip(x, y)]

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_dense(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(x, y)
       ans = list(y)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(w.vals[i], ans[p])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_dense_sparse(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos, seed=10
           )
       v = data.container_dat(data.randn)(
              self.backend, inds, shapes, seed=20
           )

       x = w.container
       y = w.container.withdata(v.arrays)

       out = tc.addgrads(y, x)
       ans = list(y)

       for i, p in enumerate(pos):
           ans[p] = tn.addgrads(ans[p], w.vals[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, xpos, ypos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [0,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [0,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0], [1, 0]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [0,  ], [2, 0]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ], [2, 0]],
   ]) 
   def test_addgrads_sparse_sparse(self, shapes, inds, xpos, ypos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, xpos, seed=10
           )
       v = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, ypos, seed=20
           )

       xvals = w.vals
       yvals = [w.tensorspaces[p].fillwith(v.arrays[p]) for p in ypos]

       x = SparseGrad(w.space, xpos, xvals)
       y = SparseGrad(w.space, ypos, yvals)

       out = tc.addgrads(x, y)
       ans = list(w.dense)

       for i, p in enumerate(ypos):
           ans[p] = tn.addgrads(ans[p], yvals[i])

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_dense(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(x, y)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_dense_null(self, shapes, inds):

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(y, x)
       ans = list(w.tensors)

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"]],
   ]) 
   def test_addgrads_null_null(self, shapes, inds):

       w = data.null_container_dat(data.randn)(
              self.backend, inds, shapes
           )

       x = NullGrad(w.space)
       y = NullGrad(w.space)

       out = tc.addgrads(x, y)
       assert out == x


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_null_sparse(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(y, x)
       ans = x.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)


   @pytest.mark.parametrize("shapes, inds, pos", [ 
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [1,  ]],
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk", "klm", "mqlj"], [2, 0]],
   ]) 
   def test_addgrads_sparse_null(self, shapes, inds, pos):

       w = data.sparse_container_dat(data.randn)(
              self.backend, inds, shapes, pos
           )

       x = w.container
       y = w.container.tonull()

       out = tc.addgrads(x, y)
       ans = x.todense()

       for outi, ansi in zip(out, ans):
           assert tn.allclose(outi, ansi)




###############################################################################
###                                                                         ###
###  Container grads                                                        ###
###                                                                         ###
###############################################################################


# --- Container grads ------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestGradsContainer:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]],   
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_getitem(self, shapes, inds):

       positions = {
          1: [0,       slice(0,1),                       ],
          2: [0, 1,    slice(0,1), slice(0,2), slice(1,2)],
          3: [0, 1, 2, slice(0,1), slice(0,2), slice(1,2), 
                       slice(1,3), slice(2,3), slice(0,3)],
       }[len(shapes)]


       def fun(x, pos):
           return x[pos]

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:
           assert_grad(fun, submode="container")(w.container, pos)


   @pytest.mark.parametrize("shapes, inds", [
      [[(3,4,6),                   ], ["ijk",               ]],   
      [[(3,4,6), (6,2,5)           ], ["ijk",  "klm",       ]], 
      [[(3,4,6), tuple(), (6,2,5)  ], ["ijk",  "",    "klm" ]],  
      [[(3,4,6), (6,2,5), (5,7,2,4)], ["ijk",  "klm", "mqlj"]],
   ]) 
   def test_sparsegrad(self, shapes, inds):

       positions = {
          1: [0,       slice(0,1),                       ],
          2: [0, 1,    slice(0,1), slice(0,2), slice(1,2)],
          3: [0, 1, 2, slice(0,1), slice(0,2), slice(1,2), 
                       slice(1,3), slice(2,3), slice(0,3)],
       }[len(shapes)]

       def fun(x, pos, space):
           return tc.sparsegrad(x, pos, space)

       w = data.container_dat(data.randn)(
              self.backend, inds, shapes
           )

       for pos in positions:

           x = w.tensors[pos]

           if isinstance(pos, slice):
              x = ContainerGen(x)

           assert_grad(fun, submode="container")(x, pos, w.space)





