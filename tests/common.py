#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import numpy.testing as nptest




def assert_close(x, ans, rtol=2**(-16), atol=2**(-32)):

    nptest.assert_allclose(x, ans, rtol=rtol, atol=atol) 






























































