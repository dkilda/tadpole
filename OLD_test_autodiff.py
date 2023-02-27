#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common as common
import tadpole      as td


"""
pytest autodiff/test_autodiff.py 

"""




Fun = collections.namedtuple("Fun", ["call", "args", "val", "grad"])




@pytest.fixture
def fun_001():

    def fun(x, y):
        return td.add(td.mul(td.sin(td.cos(x)), td.cos(y)), td.sin(x)) 

    x = 2.7
    y = 5.1

    val   = 0.13034543157
    grads = {
             0: -1.00397095286, 
             1: -0.72755303459,
            }

    return Fun(fun, (x,y), val, lambda adx: grads[adx])




@pytest.fixture
def fun_002():

    def fun(x, x1=3.0):
      def fun1(y):
          return td.mul(x, td.mul(y, y))
      return td.grad(fun1)(x1)

    x    = 5.0
    val  = 30.0
    grad = 6.0

    return Fun(fun, (x,), val, lambda adx: grad)




@pytest.fixture
def fun_003():

    def fun(x):
      def fun1(y):
          return td.mul(x, y)
      return td.grad(fun1)(x)

    x    = 5.0
    val  = 5.0
    grad = 1.0

    return Fun(fun, (x,), val, lambda adx: grad)




class TestAD:

   @pytest.mark.parametrize("fun", [
      "fun_001", 
      "fun_002", 
      "fun_003",
   ]) 
   def test_fun(self, fun, request):

       fun = request.getfixturevalue(fun)
       val = fun.call(*fun.args)

       assert common.allclose(val, fun.val)


   @pytest.mark.parametrize("fun, adx", [
      ("fun_001", 0), 
      ("fun_001", 1),
      ("fun_002", None), 
      ("fun_003", None),
   ])
   def test_grad(self, fun, adx, request):

       fun  = request.getfixturevalue(fun)
       grad = td.grad(fun.call, adx)(*fun.args)

       assert common.allclose(grad, fun.grad(adx))


   @pytest.mark.parametrize("fun, adx", [
      ("fun_001", 0), 
      ("fun_001", 1),
   ])
   def test_deriv(self, fun, adx, request):

       fun  = request.getfixturevalue(fun)
       grad = td.deriv(fun.call, adx)(*fun.args)

       assert common.allclose(grad, fun.grad(adx))





