#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.wrapper.operations as td




###############################################################################
###                                                                         ###
###  VJP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

ad.vjpmap.add(td.neg, lambda g, out, x: td.neg(g))
ad.vjpmap.add(td.sin, lambda g, out, x: td.mul(g, td.cos(x)))
ad.vjpmap.add(td.cos, lambda g, out, x: td.neg(td.mul(g, td.sin(x))))




# --- Array operations: binary (for gradient accumulation) ------------------ #

ad.vjpmap.add(td.addgrads, lambda g, out, x, y: g, 
                           lambda g, out, x, y: g)




# --- Array operations: binary ---------------------------------------------- #

ad.vjpmap.add(td.add, lambda g, out, x, y: g, 
                      lambda g, out, x, y: g)

ad.vjpmap.add(td.sub, lambda g, out, x, y: g, 
                      lambda g, out, x, y: td.neg(g))

ad.vjpmap.add(td.mul, lambda g, out, x, y: td.mul(y, g), 
                      lambda g, out, x, y: td.mul(x, g))













