#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.wrapper.operations as op




###############################################################################
###                                                                         ###
###  JVP's of array operations                                              ###
###                                                                         ###
###############################################################################


# --- Array operations: unary ----------------------------------------------- #

ad.makejvp(op.neg, lambda g, out, x: op.neg(g))
ad.makejvp(op.sin, lambda g, out, x: op.mul(g, op.cos(x)))
ad.makejvp(op.cos, lambda g, out, x: op.neg(op.mul(g, op.sin(x))))




# --- Array operations: binary (for gradient accumulation) ------------------ #

ad.makejvp(op.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array operations: binary ---------------------------------------------- #

ad.makejvp(op.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)

ad.makejvp(op.sub, lambda g, out, x, y: g, 
                   lambda g, out, x, y: op.neg(g))

ad.makejvp(op.mul, lambda g, out, x, y: op.mul(y, g), 
                   lambda g, out, x, y: op.mul(x, g))




