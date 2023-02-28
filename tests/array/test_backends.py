#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.backends as backends




###############################################################################
###                                                                         ###
###  Backend registry and its access ports                                  ###
###                                                                         ###
###############################################################################


# --- Backend registry and its access ports --------------------------------- #

class TestRegistry:

   @pytest.mark.parametrize("backend", ["numpy"])
   def test_get_by_name(self, backend):

       typ = {
              "numpy": backends.numpy.NumpyBackend, 
              "torch": backends.torch.TorchBackend, 
             }[backend]

       out = backends.get(backend)

       assert isinstance(out, typ)


   @pytest.mark.parametrize("backend", ["numpy"])
   def test_get_by_obj(self, backend):

       x = {
            "numpy": backends.numpy.NumpyBackend, 
            "torch": backends.torch.TorchBackend, 
           }[backend]()

       out = backends.get(x)

       assert out is x


   def test_get_default(self):

       assert backends.get(None) == backends.get("numpy")


   def test_get_error(self):

       try:
          out = backends.get(["numpy"])
       except ValueError:
          assert True
       else:
          assert False


   @pytest.mark.parametrize("backend", ["numpy"])
   def test_get_str(self, backend):

       x = data.array_dat(data.randn)(backend, (2,3,4))
       assert backends.get_str(x.data) == backend





















