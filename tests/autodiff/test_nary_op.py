#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

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

class TestNaryOp:

   @pytest.mark.parametrize("args, adx", [
      [(fake.NodeLike(), fake.NodeLike(), fake.Value()),    0],
      [(fake.NodeLike(), fake.Value(),    fake.NodeLike()), 1],
   ])  
   def test_call(self, args, adx):

       x   = args[adx]
       x1  = fake.NodeLike()
       out = fake.NodeLike()

       args1      = list(args)
       args1[adx] = x1

       unary_op = fake.Op(fake.Fun(x1, x))
       fun      = fake.Fun(out, *args1)
       proxy    = fake.ArgProxy(
                                insert=fake.Fun(args1, args, x1), 
                                extract=fake.Fun(x, args)
                               )

       nary_op = tdnary.NaryOp(unary_op, fun, proxy)
       assert nary_op(*args) == out



   @pytest.mark.parametrize("adx, proxytype", [
      [0,     "SINGULAR"],
      [1,     "SINGULAR"],
      [(0,1), "PLURAL"],
   ])  
   def test_make_nary_op(self, adx, proxytype):

       proxy = {
                "SINGULAR": tdutil.SingularArgProxy,
                "PLURAL":   tdutil.PluralArgProxy,
               }[proxytype](adx)

       def fun(*args): 
           return fake.Value()

       def unary_op(fun, x):
           return fake.Value()

       nary_op = tdnary.NaryOp(unary_op, fun, proxy)
       assert tdnary.make_nary_op(unary_op)(fun, adx) == nary_op


   def test_make_nary_op_001(self):

       def fun(*args): 
           return fake.Value()

       def unary_op(fun, x):
           return fake.Value()

       nary_op = tdnary.NaryOp(unary_op, fun, tdutil.SingularArgProxy(0))

       print("\nTEST-1: ", tdnary.make_nary_op(unary_op)(fun))
       print("\nTEST-2: ", nary_op)

       assert tdnary.make_nary_op(unary_op)(fun) == nary_op




























































