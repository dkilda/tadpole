#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.adjoints as tda

from tadpole.autodiff.wrapper.functions import (
   add, 
   sub, 
   mul, 
   sin, 
   cos, 
   neg, 
   floor, 
   equals,
)




###############################################################################
###                                                                         ###
###  Creating JVPs                                                          ###
###                                                                         ###
###############################################################################


# --- JVPs ------------------------------------------------------------------ #

tda.jvpmap.add(add, lambda g, out, x, y: g, 
                    lambda g, out, x, y: g)

tda.jvpmap.add(sub, lambda g, out, x, y: g, 
                    lambda g, out, x, y: neg(g))

tda.jvpmap.add(mul, lambda g, out, x, y: mul(y, g), 
                    lambda g, out, x, y: mul(x, g))

tda.jvpmap.add(neg, lambda g, out, x: neg(g))
tda.jvpmap.add(sin, lambda g, out, x: mul(g, cos(x)))
tda.jvpmap.add(cos, lambda g, out, x: neg(mul(g, sin(x))))

























































