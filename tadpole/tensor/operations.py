#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as ar

import tadpole.tensor.core    as core
import tadpole.tensor.funcall as fn


from tadpole.tensor.index import (
   Index, 
   Indices,
   shapeof, 
   sizeof,
)


from tadpole.tensor.core import (
   typecast_unary,
   typecast_binary,
)




###############################################################################
###                                                                         ###
###  Definitions of matching, typecasting and broadcasting operations       ###
###                                                                         ###
###############################################################################


# --- Typecast -------------------------------------------------------------- #

def typecast(x, y):

    if iscomplex(x) and not iscomplex(y):
       return real(x)

    if not iscomplex(x) and iscomplex(y):
       return x + 0j

    return x




# --- Upcast (adding new indices to match tensor shapes) -------------------- # 



def unreduce_grad(x, target, inds=None): # TODO this seems exclusively for grads

    def fun(g):

        g1 = extend(g, target, inds)
        x1 = extend(x, target, inds)

        mask = core.isequal(target, x1)

        return g1 * mask / sumover(mask, inds)

    return fun



def extend(x, inds):

    def fun(xdata, xinds):

        xinds = xinds.add(*inds)
        xdata = ar.broadcast_to(xdata, xinds.shape)

        return core.Tensor(xdata, xinds)

    return fn.Args(x).pluginto(fn.Transform(fun))



def typematch(x, y):

    if iscomplex(x) and not iscomplex(y):
       return real(x)

    if not iscomplex(x) and iscomplex(y):
       return x + 0j

    return x



def downmatch(x, y): 

    while ndim(x) > ndim(y): # TODO which inds to sumover? 
       x = sumover(x, ind)

    return 



def upmatch(x, y, inds):

    if y.ndim == 0:
       return x

    


    extend(x, )



# --- Downcast (removing indices to match tensor shapes) -------------------- # 




# --- Unreduce (reverse of reduction by upcasting a tensor) ----------------- # 




###############################################################################
###                                                                         ###
###  Definitions of non-differentiable tensor operations                    ###
###                                                                         ###
###############################################################################


# --- Tensor value methods -------------------------------------------------- #

@ad.nondifferentiable
def allof(x, inds=None, **opts):

    def fun(data, axis): 
        return ar.allof(data, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.nondifferentiable
def anyof(x, inds=None, **opts):

    def fun(data, axis): 
        return ar.anyof(data, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.nondifferentiable
def count_nonzero(x, inds=None, **opts):

    def fun(data, axis): 
        return ar.count_nonzero(data, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.nondifferentiable
def sign(x, **opts):

    def fun(data): 
        return ar.sign(data, **opts)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.nondifferentiable
def put(x, pos, vals, accumulate=False): 

    def fun(data):
        return ar.put(data, pos, vals, accumulate=accumulate)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.nondifferentiable
@typecast_unary
def iscomplex(x):

    def fun(data):
        return ar.iscomplex(data)

    return fn.Args(x).pluginto(fn.Extract(fun))  




###############################################################################
###                                                                         ###
###  Definitions of differentiable tensor operations                        ###
###                                                                         ###
###############################################################################


# --- Tensor shape methods -------------------------------------------------- #

@ad.differentiable
def reindex(x, indmap):

    def fun(inds):

        outinds = list(inds)

        for i, ind in enumerate(inds):
            try:
                outinds[i] = indmap[ind]
            except KeyError:
                pass

        return Indices(*outinds) 

    return fn.Args(x).pluginto(fn.Reindex(fun))




@ad.differentiable
def fuse(x, fusemap):

    if isinstance(fusemap, dict):
       fusemap = fusemap.items()

    def fun(inds):

        for inp, out in fusemap:

            inp = inds.map(*inp)

            if not isinstance(out, Index):
               out = Index(out, sizeof(*inp))

            assert sizeof(*inp) == sizeof(out), (
               f"fuse: "
               f"sizes of input {inp} and output {out} indices "
               f"must match, but {sizeof(*inp)} != {sizeof(out)}."
            )

            inds = inds.remove(*inp).add(out)

        return inds

    return fn.Args(x).pluginto(fn.Reshape(fun))




@ad.differentiable
def split(x, splitmap):

    if isinstance(splitmap, dict):
       splitmap = splitmap.items()

    def fun(inds):

        for inp, out in splitmap:

            inp, = inds.map(inp)

            assert sizeof(inp) == sizeof(*out), (
               f"split: "
               f"sizes of input {inp} and output {out} indices "
               f"must match, but {sizeof(inp)} != {sizeof(*out)}."
            )

            axis, = inds.axes(inp)
            inds  = inds.remove(inp).add(*out, axis=axis)

        return inds

    return fn.Args(x).pluginto(fn.Reshape(fun))




@ad.differentiable
def transpose(x, *order):

    def fun(data, inds):

        order = inds.map(*order)

        assert set(inds) == set(order),
           f"transpose: "
           f"input and output must contain the same set of indices, "
           f"but input {inds} does not match output {order}."

        outinds = Indices(*order)
        outdata = ar.transpose(data, inds.axes(*order))

        return core.Tensor(data, order)

    return fn.Args(x).pluginto(fn.Transform(fun))




@ad.differentiable
def squeeze(x):

    def fun(data, inds):

        singletons = (ind for ind in inds if len(ind) == 1)

        outinds = inds.remove(*singletons)
        outdata = ar.squeeze(data, inds.axes(*singletons))

        return core.Tensor(outdata, outinds)

    return fn.Args(x).pluginto(fn.Transform(fun))




@ad.differentiable
def unsqueeze(x, names):

    def fun(data, inds):

        singletons = (Index(name) for name in names)

        outinds = inds.add(*singletons)
        outdata = ar.unsqueeze(data, outinds.axes(*singletons))

        return core.Tensor(outdata, outinds)

    return fn.Args(x).pluginto(fn.Transform(fun))




@ad.differentiable
def sumover(x, inds=None, dtype=None, **opts):

    def fun(data, axis):
        return ar.sumover(data, axis, dtype, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))




@ad.differentiable
def cumsum(x, inds=None, dtype=None, **opts):

    def fun(data, axis):
        return ar.cumsum(data, axis, dtype, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))




# --- Tensor value methods -------------------------------------------------- #

@ad.differentiable
def amax(x, inds=None, **opts):

    def fun(data, axis): 
        return ar.max(data, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.differentiable
def amin(x, inds=None, **opts):

    def fun(data, axis): 
        return ar.min(data, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.differentiable
def absolute(x, **opts):

    def fun(data): 
        return ar.absolute(data, **opts)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
def flip(x, flipinds=None):

    def fun(data, inds): 

        axes = None
        if flipinds is not None:
           axes = inds.axes(*flipinds)

        outdata = ar.flip(data, axes)
        return core.Tensor(outdata, inds)

    return fn.Args(x).pluginto(fn.Transform(fun))



@ad.differentiable
def clip(x, minval, maxval, **opts):

    def fun(data): 
        return ar.clip(data, minval, maxval, **opts)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_binary
def where(condition, x, y):

    def fun(cond_uv, u, v): 
        return ar.where(cond_uv, u, v)

    return fn.Args(condition, x, y).pluginto(fn.Elemwise(fun))




# --- Standard math --------------------------------------------------------- #

@ad.differentiable
@typecast_unary
def conj(x, **opts):

    def fun(v):
        return ar.conj(v, **opts)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def real(x):

    def fun(v):
        return ar.real(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))     



@ad.differentiable
@typecast_unary
def imag(x):

    def fun(v):
        return ar.imag(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))
  


@ad.differentiable
@typecast_unary
def sqrt(x):

    def fun(v):
        return ar.sqrt(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def log(x):

    def fun(v):
        return ar.log(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def exp(x):

    def fun(v):
        return ar.exp(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def sin(x):

    def fun(v):
        return ar.sin(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def cos(x):

    def fun(v):
        return ar.cos(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def tan(x):

    def fun(v):
        return ar.tan(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arcsin(x):

    def fun(v):
        return ar.arcsin(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arccos(x):

    def fun(v):
        return ar.arccos(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arctan(x):

    def fun(v):
        return ar.arctan(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def sinh(x):

    def fun(v):
        return ar.sinh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def cosh(x):

    def fun(v):
        return ar.cosh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def tanh(x):

    def fun(v):
        return ar.tanh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arcsinh(x):

    def fun(v):
        return ar.arcsinh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arccosh(x):

    def fun(v):
        return ar.arccosh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))



@ad.differentiable
@typecast_unary
def arctanh(x):

    def fun(v):
        return ar.arctanh(v)

    return fn.Args(x).pluginto(fn.Elemwise(fun))




# --- Linear algebra: misc methods ------------------------------------------ #

@ad.differentiable
def norm(x, order=None, inds=None, **opts):

    def fun(data, axis):
        return ar.norm(data, order, axis, **opts)

    return fn.Args(x).pluginto(fn.Reduce(fun, inds))



@ad.differentiable
def expm(x):

    def fun(data):
        return ar.expm(data)

    return fn.Args(x).pluginto(fn.Elemwise(fun))
       


def htranspose(x, *order):

    return transpose(conj(x), *order)








