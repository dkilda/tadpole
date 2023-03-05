#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tadpole as td




# --- Scalar function data -------------------------------------------------- #

ScalarData = collections.namedtuple("ScalarData", [
             "fun", "args", "out", "grad"
          ])




def scalar_dat_001():

    def fun(x, y):
        return td.add(td.mul(td.sin(td.cos(x)), td.cos(y)), td.sin(x)) 

    x = td.asarray(2.7)
    y = td.asarray(5.1)

    out   = td.asarray(0.13034543157)
    grads = {
             0: td.asarray(-1.00397095286), 
             1: td.asarray(-0.72755303459),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])




def scalar_dat_002():

    x1 = td.asarray(3.0)

    def fun(x, x1=x1):
      def fun1(y):
          return td.mul(x, td.mul(y, y))
      return td.gradient(fun1)(x1)

    x    = td.asarray(5.0)
    out  = td.asarray(30.0)
    grad = td.asarray(6.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_003():

    def fun(x):
      def fun1(y):
          return td.mul(x, y)
      return td.gradient(fun1)(x)

    x    = td.asarray(5.0)
    out  = td.asarray(5.0)
    grad = td.asarray(1.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_004():

    def fun(x, y):
        return td.add(x, y) 

    x = td.asarray(2.7)
    y = td.asarray(5.1)

    out   = td.asarray(7.8)
    grads = {
             0: td.asarray(1.0), 
             1: td.asarray(1.0),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])




def scalar_dat_005():

    def fun(x):
      def fun1(y):
          return td.add(x, y)
      return td.gradient(fun1)(x)

    x    = td.asarray(5.0)
    out  = td.asarray(1.0)
    grad = td.asarray(1.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_006():

    def fun(x, y):
        return td.add(y, td.sin(x))  

    x = td.asarray(2.7)
    y = td.asarray(5.1)

    out   = td.asarray(5.52737988023) 
    grads = {
             0: td.asarray(-0.90407214201), 
             1: td.asarray(1.0),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])








