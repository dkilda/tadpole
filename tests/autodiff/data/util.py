#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.util as tdutil
import tadpole.autodiff.node as tdnode




# --- Loop iterator --------------------------------------------------------- #

LoopData = collections.namedtuple("LoopData", [
              "loop", "xs", "reversed_xs", "last", 
              "first", "next", "stop",
           ]) 




def loop_dat(): 

    xs  = (1,4,7,10,13,16,19,22)
    rxs = (13,10,7,4,1) 

    first_ = 1
    last_  = 13

    def next_(x):
        return x+3

    def stop_(x):
        return x > 15 

    loop = tdutil.Loop(first_, next_, stop_)

    return LoopData(loop, xs, rxs, last_, first_, next_, stop_)




# --- Argument proxy -------------------------------------------------------- #

ArgProxyData = collections.namedtuple("ArgProxyData", [
                  "argproxy", "adx", "x", "args", "args1",
               ]) 




def singular_argproxy_dat(adx):

    x     = fake.Value()
    args  = common.arepeat(fake.Value, 4)
    args1 = {
             0: lambda: (x,       args[1], args[2], args[3]),
             1: lambda: (args[0], x,       args[2], args[3]),
             2: lambda: (args[0], args[1], x,       args[3]),
             3: lambda: (args[0], args[1], args[2], x),
            }[adx]()

    argproxy = tdutil.SingularArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)




def plural_argproxy_dat(adx):

    x     = common.arepeat(fake.Value, len(adx))
    args  = common.arepeat(fake.Value, 4)
    args1 = {
             (1,):    lambda: (args[0], x[0],    args[2], args[3]),
             (0,1):   lambda: (x[0],    x[1],    args[2], args[3]),
             (1,3):   lambda: (args[0], x[0],    args[2], x[1]),
             (0,2,3): lambda: (x[0],    args[1], x[1],    x[2]),
             (0,3):   lambda: (x[0],    args[1], args[2], x[1]),
            }[adx]()

    argproxy = tdutil.PluralArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)


             

def singular_argproxy_dat_001():

    adx = 0

    x     = fake.Value()
    args  = tuple()
    args1 = (x,)

    argproxy = tdutil.SingularArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)




def plural_argproxy_dat_001():

    adx = (0,1)

    x     = common.arepeat(fake.Value, len(adx))
    args  = tuple()
    args1 = tuple(x)

    argproxy = tdutil.PluralArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)    




