#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.array.operations as op



###############################################################################
###                                                                         ###
###  VJP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

ad.vjpmap.add(op.neg, lambda g, out, x: op.neg(g))
ad.vjpmap.add(op.sin, lambda g, out, x: op.mul(g, op.cos(x)))
ad.vjpmap.add(op.cos, lambda g, out, x: op.neg(op.mul(g, op.sin(x))))




# --- Array operations: binary ---------------------------------------------- #

ad.vjpmap.add(op.add, lambda g, out, x, y: g, 
                      lambda g, out, x, y: g)

ad.vjpmap.add(op.sub, lambda g, out, x, y: g, 
                      lambda g, out, x, y: op.neg(g))

ad.vjpmap.add(op.mul, lambda g, out, x, y: op.mul(y, g), 
                      lambda g, out, x, y: op.mul(x, g))





































