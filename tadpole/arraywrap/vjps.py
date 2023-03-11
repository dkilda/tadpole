#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util     as util
import tadpole.autodiff as ad

import tadpole.arraywrap.operations as op


# TODO remember unbroadcast/match function!




###############################################################################
###                                                                         ###
###   VJP's of differentiable array operations                              ###
###                                                                         ###
###############################################################################


# --- Array member methods: arithmetics and element access ------------------ # 

# FIXME what happens if x = Node? We must wrap sparse() with @differentiable!
ad.makevjp(op.getitem, lambda g, out, x, idx: x.space().sparse(idx, out.item())

# TODO also wrap diag() with @differentiable!



ad.makevjp(op.neg, lambda g, out, x: -g)


ad.makevjp(op.add, lambda g, out, x, y: g, 
                   lambda g, out, x, y: g)


ad.makevjp(op.sub, lambda g, out, x, y:  g, 
                   lambda g, out, x, y: -g)


ad.makevjp(op.mul, lambda g, out, x, y: y * g, 
                   lambda g, out, x, y: x * g)


ad.makevjp(op.div, lambda g, out, x, y:  g / y, 
                   lambda g, out, x, y: -g * x / y**2)


ad.makevjp(op.power, 
   lambda g, out, x, y: g * y   * (x ** op.where(y, y-1, 1.)),
   lambda g, out, x, y: g * out * op.log(op.where(x, x, 1.))
)   




# --- Array methods: for gradient accumulation ------------------------------ #

ad.makevjp(op.addgrads, lambda g, out, x, y: g, 
                        lambda g, out, x, y: g)




# --- Array shape methods --------------------------------------------------- #

ad.makevjp(op.reshape, 
   lambda g, out, x, shape: op.reshape(g, op.shape(x))
)



def argsort(xs):

    return sorted(range(len(xs)), key=xs.__getitem__)




ad.makevjp(op.transpose, 
   lambda g, out, x, axes: op.transpose(g, argsort(axes)) # FIXME define your own argsort, cuz we're only
)                                                         #       applying to axes, not actual array


ad.makevjp(op.moveaxis,
   lambda g, out, x, source, destination: op.moveaxis(g, destination, source)
)


ad.makevjp(op.squeeze,
   lambda g, out, x, axis: op.reshape(g, op.shape(x))
) # FIXME enable kwargs in Envelope/Pack/AdjointOp/etc?


ad.makevjp(op.unsqueeze,
   lambda g, out, x, axis: op.reshape(g, op.shape(x))
)
 



# --- Array value methods --------------------------------------------------- #

ad.makevjp(op.amax,
   lambda g, out, axis: # TODO
)


ad.makevjp(op.amin,
   lambda g, out, axis: # TODO
)


ad.makevjp(op.absolute,
   lambda g, out, x: g * op.conj(x) / out
)


ad.makevjp(op.flip,
   lambda g, out, x, axis: op.flip(g, axis)
)


ad.makevjp(op.clip,
   lambda g, out, x, minval, maxval: 
      g * op.logical_and(op.not_equal(out, minval), op.not_equal(out, maxval)) 
) 



ad.makevjp(op.where,
   None,
   lambda g, out, condition, x, y: op.where(condition, g, core.zeros(g.shape)),
   lambda g, out, condition, x, y: op.where(condition, core.zeros(g.shape), g),
)




# --- Simple math operations ------------------------------------------------ #

ad.makevjp(op.conj, lambda g, out, x: op.conj(g))


ad.makevjp(op.sqrt, lambda g, out, x: g * 0.5 * (x**-0.5))
ad.makevjp(op.log,  lambda g, out, x: g / x)
ad.makevjp(op.exp,  lambda g, out, x: out * g)


ad.makevjp(op.sin, lambda g, out, x:  g * op.cos(x))
ad.makevjp(op.cos, lambda g, out, x: -g * op.sin(x))
ad.makevjp(op.tan, lambda g, out, x:  g / (op.cos(x) ** 2))


ad.makevjp(op.arcsin, lambda g, out, x:  g / op.sqrt(1 - x**2))
ad.makevjp(op.arccos, lambda g, out, x: -g / op.sqrt(1 - x**2))
ad.makevjp(op.arctan, lambda g, out, x:  g / (1 + x**2))


ad.makevjp(op.sinh, lambda g, out, x:  g * op.cosh(x))
ad.makevjp(op.cosh, lambda g, out, x: -g * op.sinh(x))
ad.makevjp(op.tanh, lambda g, out, x:  g / (op.cosh(x) ** 2))


ad.makevjp(op.arcsinh, lambda g, out, x:  g / op.sqrt(x**2 + 1))
ad.makevjp(op.arccosh, lambda g, out, x: -g / op.sqrt(x**2 - 1))
ad.makevjp(op.arctanh, lambda g, out, x:  g / (1 - x**2))


ad.makevjp(op.sumover, 
   lambda g, out, x: # TODO, we need repeat_to_match_shape!
)

ad.makevjp(op.cumsum, 
   lambda g, out, x, axis, dtype: # TODO we need reverse_axis, grad_cumsum!
)




# --- Linear algebra: multiplication methods -------------------------------- #





# --- Linear algebra: decomposition methods --------------------------------- #


# --- Linear algebra: matrix exponential ------------------------------------ #


# --- Linear algebra: misc methods ------------------------------------------ #

ad.makevjp(op.htranspose,
   lambda g, out, x, axes: op.transpose(op.conj(g), op.argsort(axes))
) 















