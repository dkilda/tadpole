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
import tadpole.linalg   as la

import tadpole.array.backends as backends
import tadpole.tensor.engine  as tne 

import tests.linalg.fakes as fake
import tests.linalg.data  as data

from tests.common import (
   available_backends,
)

from tadpole.tensor.types import (
   Tensor, 
)

from tadpole.index import (
   Index,
   IndexGen,  
   Indices,
)




###############################################################################
###                                                                         ###
###  Truncation of singular/eigenvalue decompositions                       ###
###                                                                         ###
###############################################################################


# --- Truncation ------------------------------------------------------------ #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestTrunc:

   # --- Backend fixture --- #

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Helpers --- #

   def _relerr1(self, w, rank):

       if rank == w.rank:
          return 0

       return ar.sumover(w.S[rank:]) / ar.sumover(w.S)


   def _relerr2(self, w, rank):

       if rank == w.rank:
          return 0

       return (ar.sumover(w.S[rank:]**2) / ar.sumover(w.S**2))**0.5


   def _err1(self, w, rank):

       if rank == w.rank:
          return 0

       return ar.sumover(w.S[rank:]) 


   def _err2(self, w, rank):

       if rank == w.rank:
          return 0

       return ar.sumover(w.S[rank:]**2)**0.5


   # --- Tests --- #

   def test_null(self):

       w     = data.svd_trunc_dat(self.backend)
       trunc = la.TruncNull()

       assert trunc.rank(w.S) == w.rank
       assert trunc.error(w.S) == 0
       assert trunc.apply(w.U, w.S, w.V) == (w.U, w.S, w.V)


   @pytest.mark.parametrize("rank, rank1", [
      [0,  10],
      [1,  1],
      [2,  2],
      [3,  3],
      [5,  5],
      [8,  8],
      [10, 10]
   ])
   def test_rank(self, rank, rank1):

       w     = data.svd_trunc_dat(self.backend)
       trunc = la.TruncRank(rank)

       err  = trunc.error(w.S)
       err1 = self._relerr2(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-6, None, 9],
      [1e-5, 7,    7],
   ])
   def test_abs(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncAbs(cutoff)
       else:
            trunc = la.TruncAbs(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._relerr2(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-6, None, 10],
      [1e-5, 7,    7],
   ])
   def test_rel(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncRel(cutoff)
       else:
            trunc = la.TruncRel(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._relerr2(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-5, 7,    7],
   ])
   def test_relsum1(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncRelSum1(cutoff)
       else:
            trunc = la.TruncRelSum1(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._relerr1(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       S1 = S1 * ar.sumover(w.S) / ar.sumover(S1)

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-5, 7,    7],
   ])
   def test_relsum2(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncRelSum2(cutoff)
       else:
            trunc = la.TruncRelSum2(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._relerr2(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       S1 = S1 * (ar.sumover(w.S**2) / ar.sumover(S1**2))**0.5

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-5, 7,    7],
   ])
   def test_sum1(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncSum1(cutoff)
       else:
            trunc = la.TruncSum1(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._err1(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       S1 = S1 * ar.sumover(w.S) / ar.sumover(S1)

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)


   @pytest.mark.parametrize("cutoff, rank, rank1", [
      [1e-5, None, 8],
      [1e-5, 7,    7],
   ])
   def test_sum2(self, cutoff, rank, rank1):

       w = data.svd_trunc_dat(self.backend)

       if   rank is None:
            trunc = la.TruncSum2(cutoff)
       else:
            trunc = la.TruncSum2(cutoff, rank)

       err  = trunc.error(w.S)
       err1 = self._err2(w, rank1)

       U,  S,  V  = trunc.apply(w.U, w.S, w.V)
       U1, S1, V1 = w.U[:, :rank1], w.S[:rank1], w.V[:rank1, :]

       S1 = S1 * (ar.sumover(w.S**2) / ar.sumover(S1**2))**0.5

       assert trunc.rank(w.S) == rank1
       assert ar.allclose(err, err1)
       assert ar.allclose(U,   U1)
       assert ar.allclose(S,   S1)
       assert ar.allclose(V,   V1)












