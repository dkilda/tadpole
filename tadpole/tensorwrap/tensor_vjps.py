#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




###############################################################################
###                                                                         ###
###   VJP's of differentiable tensor operations                             ###
###                                                                         ###
###############################################################################


# --- Tensor member methods: arithmetics and element access ----------------- # 

ad.makevjp(tn.neg, lambda g, out, x: -g)


ad.makevjp(tn.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g
)


ad.makevjp(tn.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g
)


ad.makevjp(tn.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g
)


ad.makevjp(tn.div, lambda g, out, x, y:  g / y,       
                   lambda g, out, x, y: -g * x / y**2
)




# --- Tensor methods: for gradient accumulation ----------------------------- #

ad.makevjp(tn.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g
)




# --- Simple math operations ------------------------------------------------ #

ad.makevjp(tn.sin, lambda g, out, x:  g * tn.cos(x))
ad.makevjp(tn.cos, lambda g, out, x: -g * tn.sin(x))





"""
# --- Tensor member methods: arithmetics and element access ----------------- # 

ad.makevjp(tn.getitem, lambda g, out, x, idx: x.space().sparse(idx, out.item()))



ad.makevjp(tn.add, lambda g, out, x, y: op.match(g, x), 
                   lambda g, out, x, y: op.match(g, y)
)


ad.makevjp(tn.sub, lambda g, out, x, y: op.match( g, x), 
                   lambda g, out, x, y: op.match(-g, y)
)


ad.makevjp(tn.mul, lambda g, out, x, y: op.match(y * g, x), 
                   lambda g, out, x, y: op.match(x * g, y)
)


ad.makevjp(tn.div, lambda g, out, x, y: op.match( g / y,        x), 
                   lambda g, out, x, y: op.match(-g * x / y**2, y)
)



ad.makevjp(tn.power, 
   lambda g, out, x, y: op.match(g * y   * (x ** tn.where(y, y-1, 1.)), x),
   lambda g, out, x, y: op.match(g * out * tn.log(tn.where(x, x, 1.)),  y)
)   




# --- Tensor methods: for gradient accumulation ----------------------------- #

ad.makevjp(tn.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Tensor shape methods -------------------------------------------------- #

ad.makevjp(tn.reshape, 
   lambda g, out, x, shape: tn.reshape(g, tn.shape(x))
)


ad.makevjp(tn.transpose, 
   lambda g, out, x, axes: tn.transpose(g, util.argsort(axes)) 
)


ad.makevjp(tn.moveaxis,
   lambda g, out, x, source, destination: tn.moveaxis(g, destination, source)
)


ad.makevjp(tn.squeeze,
   lambda g, out, x, axis: tn.reshape(g, tn.shape(x))
) 


ad.makevjp(tn.unsqueeze,
   lambda g, out, x, axis: tn.reshape(g, tn.shape(x))
)
 



# --- Tensor value methods -------------------------------------------------- #

ad.makevjp(tn.amax,
   lambda g, out, x, axis=None, **opts: op.unreduce(g, x, axis=axis)
)


ad.makevjp(tn.amin,
   lambda g, out, x, axis=None, **opts: op.unreduce(g, x, axis=axis)
)


ad.makevjp(tn.absolute,
   lambda g, out, x: g * tn.conj(x) / out
)


ad.makevjp(tn.flip,
   lambda g, out, x, axis: tn.flip(g, axis)
)


ad.makevjp(tn.clip,
   lambda g, out, x, minval, maxval: 
      g * tn.logical_and(tn.notequal(out, minval), tn.notequal(out, maxval)) 
) 



ad.makevjp(tn.where,
   None,
   lambda g, out, condition, x, y: tn.where(condition, g, core.zeros(g.shape)),
   lambda g, out, condition, x, y: tn.where(condition, core.zeros(g.shape), g),
)




# --- Simple math operations ------------------------------------------------ #

ad.makevjp(tn.conj, lambda g, out, x: tn.conj(g))


ad.makevjp(tn.sqrt, lambda g, out, x: g * 0.5 * (x**-0.5))
ad.makevjp(tn.log,  lambda g, out, x: g / x)
ad.makevjp(tn.exp,  lambda g, out, x: out * g)


ad.makevjp(tn.sin, lambda g, out, x:  g * tn.cos(x))
ad.makevjp(tn.cos, lambda g, out, x: -g * tn.sin(x))
ad.makevjp(tn.tan, lambda g, out, x:  g / (tn.cos(x) ** 2))


ad.makevjp(tn.arcsin, lambda g, out, x:  g / tn.sqrt(1 - x**2))
ad.makevjp(tn.arccos, lambda g, out, x: -g / tn.sqrt(1 - x**2))
ad.makevjp(tn.arctan, lambda g, out, x:  g / (1 + x**2))


ad.makevjp(tn.sinh, lambda g, out, x:  g * tn.cosh(x))
ad.makevjp(tn.cosh, lambda g, out, x: -g * tn.sinh(x))
ad.makevjp(tn.tanh, lambda g, out, x:  g / (tn.cosh(x) ** 2))


ad.makevjp(tn.arcsinh, lambda g, out, x:  g / tn.sqrt(x**2 + 1))
ad.makevjp(tn.arccosh, lambda g, out, x: -g / tn.sqrt(x**2 - 1))
ad.makevjp(tn.arctanh, lambda g, out, x:  g / (1 - x**2))



def vjp_sumover(g, out, x, axis=None, **opts): 

    return op.extend(g, x, axis)


def vjp_cumsum(g, out, x, axis=None):

    g1 = tn.flip(tn.cumsum(tn.flip(g, axis), axis), axis) 

    if axis:
       return g1

    return tn.reshape(g1, x.shape)


ad.makevjp(tn.sumover, vjp_sumover) 
ad.makevjp(tn.cumsum,  vjp_cumsum)




# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #

"""








    
    









