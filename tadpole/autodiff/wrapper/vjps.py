#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.autodiff.adjoint_factory import VjpFactory




###############################################################################
###                                                                         ###
###  Creating VJPs                                                          ###
###                                                                         ###
###############################################################################


# --- VJPs ------------------------------------------------------------------ #

VjpFactory.add(add, lambda g, out, x, y: g, 
                    lambda g, out, x, y: g)

VjpFactory.add(sub, lambda g, out, x, y: g, 
                    lambda g, out, x, y: neg(g))

VjpFactory.add(mul, lambda g, out, x, y: mul(y, g), 
                    lambda g, out, x, y: mul(x, g))

VjpFactory.add(neg, lambda g, out, x: neg(g))
VjpFactory.add(sin, lambda g, out, x: mul(g, cos(x)))
VjpFactory.add(cos, lambda g, out, x: neg(mul(g, sin(x))))





















































