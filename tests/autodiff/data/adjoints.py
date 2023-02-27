#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.map_adjoints as adj
import tadpole.autodiff.map_jvp      as jvpmap
import tadpole.autodiff.map_vjp      as vjpmap
import tadpole.util                  as util




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

    netjvpfun = jvpmap.NetJvpFun(concat_jvpfun)
    _jvpmap   = jvpmap.JvpMap()

    return JvpMapData(_jvpmap, netjvpfun, 
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

    netvjpfun = vjpmap.NetVjpFun(concat_vjpfun)
    _vjpmap   = vjpmap.VjpMap()

    return VjpMapData(_vjpmap, netvjpfun, 
                      concat_vjpfun, vjpfuns, fun, 
                      out, adxs, grad, args, outputs)
























































