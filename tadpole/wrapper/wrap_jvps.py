#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.wrapper.operations as td




###############################################################################
###                                                                         ###
###  JVP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

ad.jvpmap.add(td.neg, lambda g, out, x: td.neg(g))
ad.jvpmap.add(td.sin, lambda g, out, x: td.mul(g, td.cos(x)))
ad.jvpmap.add(td.cos, lambda g, out, x: td.neg(td.mul(g, td.sin(x))))




# --- Array operations: binary (for gradient accumulation) ------------------ #

ad.jvpmap.add(td.addgrads, lambda g, out, x, y: g, 
                           lambda g, out, x, y: g)




# --- Array operations: binary ---------------------------------------------- #

ad.jvpmap.add(td.add, lambda g, out, x, y: g, 
                      lambda g, out, x, y: g)

ad.jvpmap.add(td.sub, lambda g, out, x, y: g, 
                      lambda g, out, x, y: td.neg(g))

ad.jvpmap.add(td.mul, lambda g, out, x, y: td.mul(y, g), 
                      lambda g, out, x, y: td.mul(x, g))




