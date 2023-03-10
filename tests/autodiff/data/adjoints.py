#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util                as util
import tadpole.autodiff.adjointmap as adj




###############################################################################
###                                                                         ###
###  Common code for handling adjoint functions (both JVP and VJP)          ### 
###  from the input.                                                        ###
###                                                                         ###
###############################################################################

# --- JVP map data ---------------------------------------------------------- #

JvpMapData = collections.namedtuple("JvpMapData", [
                 "jvpmap", "netjvpfun", 
                 "concat_jvpfun", "jvpfuns", "fun", 
                 "out", "adxs", "grads", "args", "outputs"
             ]) 




def jvpmap_dat(valency=1):

    out     = fake.Value()
    adxs    = range(valency)
    grads   = arepeat(fake.Value, valency)
    args    = arepeat(fake.Value, valency)
    outputs = arepeat(fake.Value, valency)

    jvpfuns = [fake.Fun(outputs[adx], grads[adx], out, *args) 
                    for adx in adxs]

    jvpfuns_for_concat = [fake.Fun(outputs[adx], grads[adx], adx, out, *args) 
                               for adx in adxs]

    def concat_jvpfun(g, adx, out, *args):
        return jvpfuns_for_concat[adx](g, adx, out, *args)

    def fun(*args):
        return fake.Value()

    netjvpfun = adj.NetJvpFun(concat_jvpfun)
    jvpmap    = adj.JvpMap()

    return JvpMapData(jvpmap, netjvpfun, 
                      concat_jvpfun, jvpfuns, fun, 
                      out, adxs, grads, args, outputs)




# --- VJP map data ---------------------------------------------------------- #

VjpMapData = collections.namedtuple("VjpMapData", [
                 "vjpmap", "netvjpfun", 
                 "concat_vjpfun", "vjpfuns", "fun", 
                 "out", "adxs", "grad", "args", "outputs"
             ]) 




def vjpmap_dat(valency=1):

    grad    = fake.Value()
    out     = fake.Value()
    adxs    = range(valency)
    args    = arepeat(fake.Value, valency)
    outputs = arepeat(fake.Value, valency)

    vjpfuns = [fake.Fun(outputs[adx], grad, out, *args) 
                    for adx in adxs]

    vjpfuns_for_concat = [fake.Fun(outputs[adx], grad, adx, out, *args) 
                               for adx in adxs]

    def concat_vjpfun(g, adx, out, *args):
        return vjpfuns_for_concat[adx](g, adx, out, *args)

    def fun(*args):
        return fake.Value()

    netvjpfun = adj.NetVjpFun(concat_vjpfun)
    vjpmap    = adj.VjpMap()

    return VjpMapData(vjpmap, netvjpfun, 
                      concat_vjpfun, vjpfuns, fun, 
                      out, adxs, grad, args, outputs)




