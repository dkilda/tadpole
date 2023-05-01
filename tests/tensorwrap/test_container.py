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

import tests.tensorwrap.fakes as fake
import tests.tensorwrap.data  as data

import tadpole.tensorwrap.container as tnc


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
)



###############################################################################
###                                                                         ###
###  General container                                                      ###
###                                                                         ###
###############################################################################


# --- General container ----------------------------------------------------- #

@pytest.mark.parametrize("current_backend", available_backends, indirect=True)
class TestContainerGen:

   @pytest.fixture(autouse=True)
   def request_backend(self, current_backend):

       self._backend = current_backend


   @property
   def backend(self):

       return self._backend


   # --- Grad methods --- #



   # --- Element access --- #



   # --- Container methods --- #















