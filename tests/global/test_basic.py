#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import data    as data
import tadpole as td

from tadpole.array.core import allequal, allclose



# --- Basic AD tests -------------------------------------------------------- #

class TestScalar:

   @pytest.mark.parametrize("scalardat", [ 
      data.scalar_dat_001, 
      data.scalar_dat_002, 
      data.scalar_dat_003, 
      #data.scalar_dat_004, 
      #data.scalar_dat_005, 
      #data.scalar_dat_006,
   ]) 
   def test_fun(self, scalardat):

       w   = scalardat()
       out = w.fun(*w.args)

       assert allclose(out, w.out)


   @pytest.mark.parametrize("scalardat, adx", [
      [data.scalar_dat_001, 0], 
      [data.scalar_dat_001, 1],
      #[data.scalar_dat_004, 0], 
      #[data.scalar_dat_004, 1],
      #[data.scalar_dat_006, 0], 
      #[data.scalar_dat_006, 1],
      [data.scalar_dat_002, None], 
      [data.scalar_dat_003, None],
   ])
   def test_gradient(self, scalardat, adx):

       w    = scalardat()
       grad = td.gradient(w.fun, adx)(*w.args)

       assert allclose(grad, w.grad(adx))


   @pytest.mark.parametrize("scalardat, adx", [
      [data.scalar_dat_001, 0], 
      [data.scalar_dat_001, 1],
   ])
   def test_derivative(self, scalardat, adx):

       w    = scalardat()
       grad = td.derivative(w.fun, adx)(*w.args)

       assert allclose(grad, w.grad(adx))








