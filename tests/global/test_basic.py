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
   ]) 
   def test_fun(self, scalardat):

       w   = scalardat()
       out = w.fun(*w.args)

       assert allclose(out, w.out)
