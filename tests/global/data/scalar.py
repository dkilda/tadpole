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
        return td.sin(td.cos(x)) * td.cos(y) + td.sin(x) 

    x = td.astensor(2.7)
    y = td.astensor(5.1)

    out   = td.astensor(0.13034543157)
    grads = {
             0: td.astensor(-1.00397095286), 
             1: td.astensor(-0.72755303459),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])




def scalar_dat_002():

    x1 = td.astensor(3.0)

    def fun(x, x1=x1):
      def fun1(y):
          return x * (y * y)
      return td.gradient(fun1)(x1)

    x    = td.astensor(5.0)
    out  = td.astensor(30.0)
    grad = td.astensor(6.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_003():

    def fun(x):
      def fun1(y):
          return x * y
      return td.gradient(fun1)(x)

    x    = td.astensor(5.0)
    out  = td.astensor(5.0)
    grad = td.astensor(1.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_004():

    def fun(x, y):
        return td.add(x, y) 

    x = td.astensor(2.7)
    y = td.astensor(5.1)

    out   = td.astensor(7.8)
    grads = {
             0: td.astensor(1.0), 
             1: td.astensor(1.0),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])




def scalar_dat_005():

    def fun(x):
      def fun1(y):
          return td.add(x, y)
      return td.gradient(fun1)(x)

    x    = td.astensor(5.0)
    out  = td.astensor(1.0)
    grad = td.astensor(1.0)

    return ScalarData(fun, (x,), out, lambda adx: grad)




def scalar_dat_006():

    def fun(x, y):
        return td.add(y, td.sin(x))  

    x = td.astensor(2.7)
    y = td.astensor(5.1)

    out   = td.astensor(5.52737988023) 
    grads = {
             0: td.astensor(-0.90407214201), 
             1: td.astensor(1.0),
            }

    return ScalarData(fun, (x,y), out, lambda adx: grads[adx])








