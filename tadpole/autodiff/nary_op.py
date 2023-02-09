#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tadpole.autodiff.util import make_argproxy




###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Nary operator --------------------------------------------------------- #

class NaryOp:

   def __init__(self, unary_op, fun, argproxy):

       self._unary_op = unary_op
       self._fun      = fun
       self._argproxy = argproxy


   def __eq__(self, other):

       return all((
                   self._unary_op == other._unary_op, 
                   self._fun      == other._fun, 
                   self._argproxy == other._argproxy,
                 ))


   def __call__(self, *args, **kwargs):

       def unary_fun(x):
           return self._fun(*self._argproxy.insert(args, x), **kwargs)
            
       return self._unary_op(unary_fun, self._argproxy.extract(args))




# --- Create nary operator -------------------------------------------------- #

def make_nary_op(unary_op):

    def wrap(fun, adx=None):
        return NaryOp(unary_op, fun, make_argproxy(adx))

    return wrap











































































