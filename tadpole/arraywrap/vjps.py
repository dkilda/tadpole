#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.arraywrap.operations as op




###############################################################################
###                                                                         ###
###  VJP's of non-differentiable array operations                           ###
###                                                                         ###
###############################################################################


# --- Array member methods: basic functionality ----------------------------- #


# --- Array member methods: properties -------------------------------------- #


# --- Logical functions: array comparisons ---------------------------------- # 


# --- Array value methods --------------------------------------------------- #




###############################################################################
###                                                                         ###
###   VJP's of differentiable array operations                              ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

ad.makevjp(op.neg, lambda g, out, x: -g)


ad.makevjp(op.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)


ad.makevjp(op.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g)


ad.makevjp(op.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g)




# --- Array methods: for gradient accumulation ------------------------------ #

ad.makevjp(op.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array shape methods --------------------------------------------------- #


# --- Array value methods --------------------------------------------------- #


# --- Simple math operations ------------------------------------------------ #

ad.makevjp(op.sin, lambda g, out, x:  g * op.cos(x))
ad.makevjp(op.cos, lambda g, out, x: -g * op.sin(x))



# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #

















