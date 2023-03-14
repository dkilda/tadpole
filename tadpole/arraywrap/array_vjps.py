#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as td

import tadpole.arraywrap.operations as op




###############################################################################
###                                                                         ###
###   VJP's of differentiable array operations                              ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

ad.makevjp(td.getitem, lambda g, out, x, idx: x.space().sparse(idx, out.item()))



ad.makevjp(td.add, lambda g, out, x, y: op.match(g, x), 
                   lambda g, out, x, y: op.match(g, y)
)


ad.makevjp(td.sub, lambda g, out, x, y: op.match( g, x), 
                   lambda g, out, x, y: op.match(-g, y)
)


ad.makevjp(td.mul, lambda g, out, x, y: op.match(y * g, x), 
                   lambda g, out, x, y: op.match(x * g, y)
)


ad.makevjp(td.div, lambda g, out, x, y: op.match( g / y,        x), 
                   lambda g, out, x, y: op.match(-g * x / y**2, y)
)



ad.makevjp(td.power, 
   lambda g, out, x, y: op.match(g * y   * (x ** td.where(y, y-1, 1.)), x),
   lambda g, out, x, y: op.match(g * out * td.log(td.where(x, x, 1.)),  y)
)   




# --- Array methods: for gradient accumulation ------------------------------ #

ad.makevjp(td.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array shape methods --------------------------------------------------- #

ad.makevjp(td.reshape, 
   lambda g, out, x, shape: td.reshape(g, td.shape(x))
)


ad.makevjp(td.transpose, 
   lambda g, out, x, axes: td.transpose(g, util.argsort(axes)) 
)


ad.makevjp(td.moveaxis,
   lambda g, out, x, source, destination: td.moveaxis(g, destination, source)
)


ad.makevjp(td.squeeze,
   lambda g, out, x, axis: td.reshape(g, td.shape(x))
) 


ad.makevjp(td.unsqueeze,
   lambda g, out, x, axis: td.reshape(g, td.shape(x))
)
 



# --- Array value methods --------------------------------------------------- #

ad.makevjp(td.amax,
   lambda g, out, x, axis=None, **opts: op.unreduce(g, x, axis=axis)
)


ad.makevjp(td.amin,
   lambda g, out, x, axis=None, **opts: op.unreduce(g, x, axis=axis)
)


ad.makevjp(td.absolute,
   lambda g, out, x: g * td.conj(x) / out
)


ad.makevjp(td.flip,
   lambda g, out, x, axis: td.flip(g, axis)
)


ad.makevjp(td.clip,
   lambda g, out, x, minval, maxval: 
      g * td.logical_and(td.notequal(out, minval), td.notequal(out, maxval)) 
) 



ad.makevjp(td.where,
   None,
   lambda g, out, condition, x, y: td.where(condition, g, core.zeros(g.shape)),
   lambda g, out, condition, x, y: td.where(condition, core.zeros(g.shape), g),
)




# --- Simple math operations ------------------------------------------------ #

ad.makevjp(td.conj, lambda g, out, x: td.conj(g))


ad.makevjp(td.sqrt, lambda g, out, x: g * 0.5 * (x**-0.5))
ad.makevjp(td.log,  lambda g, out, x: g / x)
ad.makevjp(td.exp,  lambda g, out, x: out * g)


ad.makevjp(td.sin, lambda g, out, x:  g * td.cos(x))
ad.makevjp(td.cos, lambda g, out, x: -g * td.sin(x))
ad.makevjp(td.tan, lambda g, out, x:  g / (td.cos(x) ** 2))


ad.makevjp(td.arcsin, lambda g, out, x:  g / td.sqrt(1 - x**2))
ad.makevjp(td.arccos, lambda g, out, x: -g / td.sqrt(1 - x**2))
ad.makevjp(td.arctan, lambda g, out, x:  g / (1 + x**2))


ad.makevjp(td.sinh, lambda g, out, x:  g * td.cosh(x))
ad.makevjp(td.cosh, lambda g, out, x: -g * td.sinh(x))
ad.makevjp(td.tanh, lambda g, out, x:  g / (td.cosh(x) ** 2))


ad.makevjp(td.arcsinh, lambda g, out, x:  g / td.sqrt(x**2 + 1))
ad.makevjp(td.arccosh, lambda g, out, x: -g / td.sqrt(x**2 - 1))
ad.makevjp(td.arctanh, lambda g, out, x:  g / (1 - x**2))



def vjp_sumover(g, out, x, axis=None, **opts): 

    return op.extend(g, x, axis)


def vjp_cumsum(g, out, x, axis=None):

    g1 = td.flip(td.cumsum(td.flip(g, axis), axis), axis) 

    if axis:
       return g1

    return td.reshape(g1, x.shape)


ad.makevjp(td.sumover, vjp_sumover) 
ad.makevjp(td.cumsum,  vjp_cumsum)




# --- Linear algebra: multiplication methods -------------------------------- #


# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #










    
    









