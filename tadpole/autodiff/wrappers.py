#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util                as util
import tadpole.autodiff.graph      as ag
import tadpole.autodiff.adjointmap as adjointmap




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




# --- Differentiable function wrap ------------------------------------------ #

def differentiable(fun):

    def envelope(*args, **kwargs):
        return ag.Envelope(*args, **kwargs)

    fun = util.return_outputs(fun)

    return ag.Differentiable(fun, envelope, _VJPMAP, _JVPMAP)




# --- Non-differentiable function wrap -------------------------------------- #

def nondifferentiable(fun):

    def envelope(*args, **kwargs):
        return ag.Envelope(*args, **kwargs)

    fun = util.return_outputs(fun)

    return ag.NonDifferentiable(fun, envelope)





