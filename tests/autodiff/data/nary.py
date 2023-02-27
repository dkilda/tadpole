#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.nary as nary
import tadpole.util          as util




###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Nary operator --------------------------------------------------------- #

NaryOpData = collections.namedtuple("NaryOpData", [
                "nary_op", "unary_op", "fun", "argproxy", "adx", 
                "args", "args1", "x", "x1", "out"
             ]) 




def nary_op_dat(args, adx):

    x   = args[adx]
    x1  = fake.NodeLike()
    out = fake.NodeLike()

    args1      = list(args)
    args1[adx] = x1

    unary_op = fake.Op(fake.Fun(x1, x))
    fun      = fake.Fun(out, *args1)
    argproxy = fake.ArgProxy(
                             insert=fake.Fun(args1, args, x1), 
                             extract=fake.Fun(x, args)
                            )   
    nary_op = nary.NaryOp(unary_op, fun, argproxy)

    return NaryOpData(nary_op, unary_op, fun, argproxy, adx, 
                      args, args1, x, x1, out)




# --- Nary operator creation ------------------------------------------------ #

NaryOpCreatorData = collections.namedtuple("NaryOpCreatorData", [
                       "nary_op", "unary_op", "fun", "argproxy", "adx", 
                    ]) 



def nary_op_creator_dat(adx, proxytype):

    argproxy = {
                "SINGULAR": nary.SingularArgProxy,
                "PLURAL":   nary.PluralArgProxy,
               }[proxytype](adx)  

    def fun(*args): 
        return fake.Value()

    def unary_op(fun, x):
        return fake.Value()

    nary_op = nary.NaryOp(unary_op, fun, argproxy)    

    return NaryOpCreatorData(nary_op, unary_op, fun, argproxy, adx)




# --- Argument proxy -------------------------------------------------------- #

ArgProxyData = collections.namedtuple("ArgProxyData", [
                  "argproxy", "adx", "x", "args", "args1",
               ]) 




def singular_argproxy_dat(adx):

    x     = fake.Value()
    args  = arepeat(fake.Value, 4)
    args1 = {
             0: lambda: (x,       args[1], args[2], args[3]),
             1: lambda: (args[0], x,       args[2], args[3]),
             2: lambda: (args[0], args[1], x,       args[3]),
             3: lambda: (args[0], args[1], args[2], x),
            }[adx]()

    argproxy = nary.SingularArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)




def plural_argproxy_dat(adx):

    x     = arepeat(fake.Value, len(adx))
    args  = arepeat(fake.Value, 4)
    args1 = {
             (1,):    lambda: (args[0], x[0],    args[2], args[3]),
             (0,1):   lambda: (x[0],    x[1],    args[2], args[3]),
             (1,3):   lambda: (args[0], x[0],    args[2], x[1]),
             (0,2,3): lambda: (x[0],    args[1], x[1],    x[2]),
             (0,3):   lambda: (x[0],    args[1], args[2], x[1]),
            }[adx]()

    argproxy = nary.PluralArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)


             

def singular_argproxy_dat_001():

    adx = 0

    x     = fake.Value()
    args  = tuple()
    args1 = (x,)

    argproxy = nary.SingularArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)




def plural_argproxy_dat_001():

    adx = (0,1)

    x     = arepeat(fake.Value, len(adx))
    args  = tuple()
    args1 = tuple(x)

    argproxy = nary.PluralArgProxy(adx)

    return ArgProxyData(argproxy, adx, x, args, args1)    






