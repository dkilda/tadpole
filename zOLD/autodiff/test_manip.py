#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.manip as tdmanip
import tests.autodiff.fakes   as fake

import tests.common.ntuple as tpl


# --- Wrapper for gradient addition ----------------------------------------- #

class TestAddGrads:

   @pytest.mark.parametrize("net_g, g", [
      [fake.CumFunReturn(), fake.CumFunReturn()],
   ])
   def test_add_grads(self, net_g, g):

       assert tdmanip.add_grads(net_g, g) == net_g + g


   @pytest.mark.parametrize("g", [fake.CumFunReturn()])
   def test_add_grads_simple(self, g):

       assert tdmanip.add_grads(None, g) == g










































