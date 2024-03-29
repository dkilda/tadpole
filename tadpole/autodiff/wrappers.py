#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util                as util
import tadpole.autodiff.nary       as nary
import tadpole.autodiff.graph      as ag
import tadpole.autodiff.grad       as ad
import tadpole.autodiff.adjointmap as adjointmap




###############################################################################
###                                                                         ###
###  Adjoint map wrappers                                                   ###
###                                                                         ###
###############################################################################


# --- A global instance of VJP map and its access ports --------------------- # 

_VJPMAP = adjointmap.VjpMap()


def makevjp(fun, *adjfuns, adxs=None):

    return _VJPMAP.add(fun, *adjfuns, adxs=adxs)


def makevjp_combo(fun, adjfun):

    return _VJPMAP.add_combo(fun, adjfun)


def makevjp_raw(fun, adjfun):

    return _VJPMAP.add_raw(fun, adjfun)




# --- A global instance of JVP map and its access ports --------------------- # 

_JVPMAP = adjointmap.JvpMap()


def makejvp(fun, *adjfuns, adxs=None):

    return _JVPMAP.add(fun, *adjfuns, adxs=adxs)


def makejvp_combo(fun, adjfun):

    return _JVPMAP.add_combo(fun, adjfun)


def makejvp_raw(fun, adjfun):

    return _JVPMAP.add_raw(fun, adjfun)




###############################################################################
###                                                                         ###
###  Differentiable and non-differentiable function wrappers                ###
###                                                                         ###
###############################################################################


# --- Differentiable function wrap ------------------------------------------ #

def differentiable(fun):

    def envelope(*args, **kwargs):
        return ag.EnvelopeArgs(*args, **kwargs)

    return ag.Differentiable(fun, envelope, _VJPMAP, _JVPMAP)




# --- Non-differentiable function wrap -------------------------------------- #

def nondifferentiable(fun):

    def envelope(*args, **kwargs):
        return ag.EnvelopeArgs(*args, **kwargs)

    return ag.NonDifferentiable(fun, envelope)




# --- Checkpointed function wrap -------------------------------------------- #

def checkpoint(fun):

    checkpointed_fun = differentiable(fun)

    @nary.nary_op
    def grad(fun, x):
        op = ad.diffop_reverse(fun, x)
        return lambda g: op.grad(g)

    def checkpointed_vjpfun(g, adx, out, *args, **kwargs):
        return grad(fun, adx)(*args, **kwargs)(g)

    makevjp_combo(checkpointed_fun, checkpointed_vjpfun)
    return checkpointed_fun




