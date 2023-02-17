#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.manip as tdmanip




# --- Wrapper for gradient addition ----------------------------------------- #

class TestAddGrads:

   @pytest.mark.parametrize("net_g, g", [
      [fake.Value(), fake.Value()],
   ])
   def test_add_grads(self, net_g, g):

       assert tdmanip.add_grads(net_g, g) == net_g + g


   @pytest.mark.parametrize("g", [fake.Value()])
   def test_add_grads_001(self, g):

       assert tdmanip.add_grads(None, g) == g





































































