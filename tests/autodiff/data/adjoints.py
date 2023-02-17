#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.adjoints.adjoints as tda
import tadpole.autodiff.adjoints.jvpmap   as tdjvp
import tadpole.autodiff.adjoints.vjpmap   as tdvjp




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
    grads   = common.arepeat(fake.Value, valency)
    args    = common.arepeat(fake.Value, valency)
    outputs = common.arepeat(fake.Value, valency)

    jvpfuns = [fake.Fun(outputs[adx], grads[adx], out, *args) 
                    for adx in adxs]

    jvpfuns_for_concat = [fake.Fun(outputs[adx], grads[adx], adx, out, *args) 
                               for adx in adxs]

    def concat_jvpfun(g, adx, out, *args):
        return jvpfuns_for_concat[adx](g, adx, out, *args)

    def fun(*args):
        return fake.Value()

    netjvpfun = tdjvp.NetJvpFun(concat_jvpfun)
    jvpmap    = tdjvp.JvpMap()

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
    args    = common.arepeat(fake.Value, valency)
    outputs = common.arepeat(fake.Value, valency)

    vjpfuns = [fake.Fun(outputs[adx], grad, out, *args) 
                    for adx in adxs]

    vjpfuns_for_concat = [fake.Fun(outputs[adx], grad, adx, out, *args) 
                               for adx in adxs]

    def concat_vjpfun(g, adx, out, *args):
        return vjpfuns_for_concat[adx](g, adx, out, *args)

    def fun(*args):
        return fake.Value()

    netvjpfun = tdvjp.NetVjpFun(concat_vjpfun)
    vjpmap    = tdvjp.VjpMap()

    return VjpMapData(vjpmap, netvjpfun, 
                      concat_vjpfun, vjpfuns, fun, 
                      out, adxs, grad, args, outputs)
























































