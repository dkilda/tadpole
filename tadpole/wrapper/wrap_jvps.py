#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util               as util
import tadpole.autodiff.map_jvp   as jvpmap
import tadpole.wrapper.operations as td




###############################################################################
###                                                                         ###
###  JVP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

jvpmap.add(td.neg, lambda g, out, x: td.neg(g))
jvpmap.add(td.sin, lambda g, out, x: td.mul(g, td.cos(x)))
jvpmap.add(td.cos, lambda g, out, x: td.neg(td.mul(g, td.sin(x))))




# --- Array operations: binary (for gradient accumulation) ------------------ #

jvpmap.add(td.addgrads, lambda g, out, x, y: g, 
                           lambda g, out, x, y: g)




# --- Array operations: binary ---------------------------------------------- #

jvpmap.add(td.add, lambda g, out, x, y: g, 
                      lambda g, out, x, y: g)

jvpmap.add(td.sub, lambda g, out, x, y: g, 
                      lambda g, out, x, y: td.neg(g))

jvpmap.add(td.mul, lambda g, out, x, y: td.mul(y, g), 
                      lambda g, out, x, y: td.mul(x, g))




