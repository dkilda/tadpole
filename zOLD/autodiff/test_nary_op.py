#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.nary_op as tdnary
import tadpole.autodiff.util    as tdutil
import tests.autodiff.fakes     as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Nary operator --------------------------------------------------------- #

class TestNaryOp:

   @pytest.fixture(autouse=True)
   def request_nary_op(self, nary_op):

       self.nary_op = nary_op


   def _setup(self, args, adx, x1):

       args1      = list(args)
       args1[adx] = x1

       return tuple(args1)

 
   @pytest.mark.parametrize("args, adx", [
      [(fake.Node(), fake.Node(),      fake.FunReturn()), 0],
      [(fake.Node(), fake.FunReturn(), fake.Node()),      1],
   ])  
   def test_call(self, args, adx):

       x     = args[adx]
       x1    = fake.Node()
       args1 = self._setup(args, adx, x1)

       ans = fake.FunReturn()
       fun = fake.Fun({args1: ans}) 

       unary_op = fake.Op((x1,), {(x,): (x1,)})
       argproxy = fake.ArgProxy(
                     insert={(args, x1): args1}, extract={args: x}) 

       op = self.nary_op(unary_op, fun, argproxy)
       assert op(*args) == ans


   @pytest.mark.parametrize("adx, proxytype", [
      [0,     tdutil.SingularArgProxy],
      [1,     tdutil.SingularArgProxy],
      [(0,1), tdutil.PluralArgProxy],
   ])
   def test_make_nary_op(self, adx, proxytype):

       fun      = fake.Fun() 
       unary_op = fake.Op()

       ans = self.nary_op(unary_op, fun, proxytype(adx))
       assert tdnary.make_nary_op(unary_op)(fun, adx) == ans


   def test_make_nary_op_simple(self):

       fun      = fake.Fun() 
       unary_op = fake.Op()

       ans = self.nary_op(unary_op, fun, tdutil.SingularArgProxy(0))
       assert tdnary.make_nary_op(unary_op)(fun) == ans








































