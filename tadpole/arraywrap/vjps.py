#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.wrapper.operations as op




###############################################################################
###                                                                         ###
###  VJP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

ad.makevjp(op.neg, lambda g, out, x: op.neg(g))
ad.makevjp(op.sin, lambda g, out, x: op.mul(g, op.cos(x)))
ad.makevjp(op.cos, lambda g, out, x: op.neg(op.mul(g, op.sin(x))))




# --- Array operations: binary (for gradient accumulation) ------------------ #

ad.makevjp(op.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array operations: binary ---------------------------------------------- #

ad.makevjp(op.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)

ad.makevjp(op.sub, lambda g, out, x, y: g, 
                   lambda g, out, x, y: op.neg(g))

ad.makevjp(op.mul, lambda g, out, x, y: op.mul(y, g), 
                   lambda g, out, x, y: op.mul(x, g))













