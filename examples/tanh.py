#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td


x = td.TensorGen(np.linspace(-7, 7, 200), (td.IndexLit("i", 200),))


def fun(x):
    return tn.tanh(x)


fun(x)



































