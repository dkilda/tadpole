#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.array    as td




def extend(x, target, axis):

    if target.ndim == 0:
       return x
      
    return unsqueeze(x, axis) + target.space().zeros()

        

def unreduce(x, target, axis=None):

    def fun(g):

        g1 = extend(g, target, axis)
        x1 = extend(x, target, axis)

        mask = equal(target, x1)
        
        return g1 * mask / op.sumover(mask, axis=axis, keepdims=True)

    return fun



def match_shape(x, target, default_axis=0):

    while ndim(x) > ndim(target):
       x = sumover(x, axis=default_axis)

    for axis, size in enumerate(target.shape):
        if size == 1:
           x = sumover(x, axis=axis, keepdims=True)

    return x



def match_type(x, target):

    if iscomplex(x) and not iscomplex(target):
       return real(x)

    if not iscomplex(x) and iscomplex(target):
       return x + 0j

    return x



def match(x, target, **opts):

    return match_type(match_shape(x, target, **opts), target)
    


def htranspose(x, axes):
 
    return transpose(conj(x), axes)















