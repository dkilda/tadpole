#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tests.array.data as data

import tadpole.array.core     as tcore
import tadpole.array.function as tfunction



###############################################################################
###                                                                         ###
###  Definition of array.                                                   ###
###                                                                         ###
###############################################################################


# --- Array ----------------------------------------------------------------- #

class TestArray:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("shape",   [(2,3,4)])
   def test_shape(self, backend, shape):

       w = data.array_dat(data.randn)("numpy", shape)
       assert w.array.shape == shape





































