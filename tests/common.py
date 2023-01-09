#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.testing as nptest




# --- Assertion for array data ---------------------------------------------- #

def assert_close(x, ans, rtol=2**(-16), atol=2**(-32)):

    nptest.assert_allclose(x, ans, rtol=rtol, atol=atol) 






























































