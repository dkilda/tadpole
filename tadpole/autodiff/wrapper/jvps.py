#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.autodiff.adjoint_factory import JvpFactory




###############################################################################
###                                                                         ###
###  Creating JVPs                                                          ###
###                                                                         ###
###############################################################################


# --- JVPs ------------------------------------------------------------------ #

JvpFactory.add(add, lambda g, out, x, y: g, 
                    lambda g, out, x, y: g)

JvpFactory.add(sub, lambda g, out, x, y: g, 
                    lambda g, out, x, y: neg(g))

JvpFactory.add(mul, lambda g, out, x, y: mul(y, g), 
                    lambda g, out, x, y: mul(x, g))

JvpFactory.add(neg, lambda g, out, x: neg(g))
JvpFactory.add(sin, lambda g, out, x: mul(g, cos(x)))
JvpFactory.add(cos, lambda g, out, x: neg(mul(g, sin(x))))

























































