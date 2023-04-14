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

import tadpole.array.backends         as backends
import tadpole.tensorwrap.tensor_vjps as tvjps

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data


from tests.tensorwrap.conftest import (
   available_backends,
)


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
###  Binary elementwise VJPs                                                ###
###                                                                         ###
###############################################################################


# --- Binary elementwise VJPs ----------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestVjpElemwiseBinary:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   def test_vjp(self): # TODO dummy test, replace it!

       backend = backends.get(self.backend)
       assert backend.name() == self.backend












