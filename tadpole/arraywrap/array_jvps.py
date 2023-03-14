#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as td




###############################################################################
###                                                                         ###
###  JVP's of differentiable array operations                               ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

ad.makejvp(td.neg, lambda g, out, x: -g)


ad.makejvp(td.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)


ad.makejvp(td.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g)


ad.makejvp(td.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g)




# --- Array methods: for gradient accumulation ------------------------------ #

ad.makejvp(td.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array shape methods --------------------------------------------------- #


# --- Array value methods --------------------------------------------------- #


# --- Simple math operations ------------------------------------------------ #

ad.makejvp(td.sin, lambda g, out, x:  g * td.cos(x))
ad.makejvp(td.cos, lambda g, out, x: -g * td.sin(x))



# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #





