#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.arraywrap.operations as op




###############################################################################
###                                                                         ###
###  JVP's of differentiable array operations                               ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

ad.makejvp(op.neg, lambda g, out, x: -g)


ad.makejvp(op.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)


ad.makejvp(op.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g)


ad.makejvp(op.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g)




# --- Array methods: for gradient accumulation ------------------------------ #

ad.makejvp(op.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array shape methods --------------------------------------------------- #


# --- Array value methods --------------------------------------------------- #


# --- Simple math operations ------------------------------------------------ #

ad.makejvp(op.sin, lambda g, out, x:  g * op.cos(x))
ad.makejvp(op.cos, lambda g, out, x: -g * op.sin(x))



# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #





