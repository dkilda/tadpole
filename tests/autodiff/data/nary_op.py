#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.nary_op as tdnary
import tadpole.autodiff.util    as tdutil




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
    nary_op = tdnary.NaryOp(unary_op, fun, argproxy)

    return NaryOpData(nary_op, unary_op, fun, argproxy, adx, 
                      args, args1, x, x1, out)




# --- Nary operator creation ------------------------------------------------ #

NaryOpCreatorData = collections.namedtuple("NaryOpCreatorData", [
                       "nary_op", "unary_op", "fun", "argproxy", "adx", 
                    ]) 



def nary_op_creator_dat(adx, proxytype):

    argproxy = {
                "SINGULAR": tdutil.SingularArgProxy,
                "PLURAL":   tdutil.PluralArgProxy,
               }[proxytype](adx)  

    def fun(*args): 
        return fake.Value()

    def unary_op(fun, x):
        return fake.Value()

    nary_op = tdnary.NaryOp(unary_op, fun, argproxy)    

    return NaryOpCreatorData(nary_op, unary_op, fun, argproxy, adx)








