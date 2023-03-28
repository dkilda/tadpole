#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

import numpy as np
import scipy

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array          as ar
import tadpole.array.space    as sp
import tadpole.array.unary    as unary
import tadpole.array.binary   as binary
import tadpole.array.nary     as nary
import tadpole.array.void     as void
import tadpole.array.backends as backends

from tests.common import (
   options,
)




###############################################################################
###                                                                         ###
###  Definition of Void Array (supports array creation)                     ###
###                                                                         ###
###############################################################################


# --- Void Array ------------------------------------------------------------ #

@pytest.mark.parametrize("current_backend", ["numpy_backend"], indirect=True)
class TestArray:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Array methods --- #

   def test_nary(self):

       w = data.narray_dat(data.randn)(self.backend)

       assert w.narray.nary() == nary.Array(w.backend)





























