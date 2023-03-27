#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array          as ar
import tadpole.array.unary    as unary
import tadpole.array.backends as backends

from tests.common import (
   options,
)



###############################################################################
###                                                                         ###
###  Helper functions                                                       ###
###                                                                         ###
###############################################################################


# --- Type cast for unary functions ----------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestTypeCast:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   @pytest.mark.parametrize("fundat",  [
       data.unary_wrappedfun_dat_001,
       data.unary_wrappedfun_dat_002,
   ])
   def test_unary(self, fundat):

       w = fundat(self.backend)
       assert ar.allclose(w.wrappedfun(*w.args), w.out)




###############################################################################
###                                                                         ###
###  Definition of Unary Array (supports unary operations)                  ###
###                                                                         ###
###############################################################################


# --- Unary Array ----------------------------------------------------------- #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend
   

   @pytest.mark.parametrize("shape", [(2,3,4)])
   @pytest.mark.parametrize("dtype", ["complex128"])
   def test_asarray(self, shape, dtype):

       x = data.array_dat(data.randn)(
              self.backend, shape, dtype=dtype, seed=1
           )

       opts = options(dtype=dtype, backend=self.backend)
       out  = unary.asarray(x.data, **opts) 

       assert out == x.array




