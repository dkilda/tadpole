#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




###############################################################################
###                                                                         ###
###  JVP's of differentiable tensor operations                              ###
###                                                                         ###
###############################################################################


# --- Tensor member methods: arithmetics and element access ----------------- # 

ad.makejvp(tn.neg, lambda g, out, x: -g)


ad.makejvp(tn.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)


ad.makejvp(tn.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g)


ad.makejvp(tn.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g)




# --- Tensor methods: for gradient accumulation ----------------------------- #

ad.makejvp(tn.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Tensor shape methods -------------------------------------------------- #


# --- Tensor value methods -------------------------------------------------- #


# --- Simple math operations ------------------------------------------------ #

ad.makejvp(tn.sin, lambda g, out, x:  g * tn.cos(x))
ad.makejvp(tn.cos, lambda g, out, x: -g * tn.sin(x))



# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #





