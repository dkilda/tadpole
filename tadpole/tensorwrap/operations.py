#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.util     as util
import tadpole.autodiff as ad
import tadpole.tensor   as tn




def extend(x, target, axis):

    if target.ndim == 0:
       return x
      
    return tn.unsqueeze(x, axis) + target.space().zeros()

        

def unreduce(x, target, axis=None):

    def fun(g):

        g1 = extend(g, target, axis)
        x1 = extend(x, target, axis)

        mask = equal(target, x1)
        
        return g1 * mask / tn.sumover(mask, axis=axis, keepdims=True)

    return fun



def match_shape(x, target, default_axis=0):

    while tn.ndim(x) > tn.ndim(target):
       x = tn.sumover(x, axis=default_axis)

    for axis, size in enumerate(target.shape):
        if size == 1:
           x = tn.sumover(x, axis=axis, keepdims=True)

    return x



def match_type(x, target):

    if tn.iscomplex(x) and not tn.iscomplex(target):
       return tn.real(x)

    if not tn.iscomplex(x) and tn.iscomplex(target):
       return x + 0j

    return x



def match(x, target, **opts):

    return match_type(match_shape(x, target, **opts), target)
    















