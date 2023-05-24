#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import tadpole.util as util


from tadpole.autodiff.types import (
   ArgProxy,
)



###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Unary operator -------------------------------------------------------- #

class UnaryOp:

   def __init__(self, op, *args, **kwargs):

       self._op     = op
       self._args   = args
       self._kwargs = kwargs


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)

       rep.val("op",     self._op)
       rep.val("args",   self._args)
       rep.val("kwargs", self._kwargs)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)

       if bool(log):

          log.val(self._op,     other._op)
          log.val(self._args,   other._args)
          log.val(self._kwargs, other._kwargs)

       return bool(log)


   def __call__(self, unary_fun, args):  

       return self._op(unary_fun, args, *self._args, **self._kwargs)




# --- Nary operator --------------------------------------------------------- #

class NaryOp:

   def __init__(self, unary_op, fun, argproxy):

       if not isinstance(unary_op, UnaryOp):
          unary_op = UnaryOp(unary_op)

       self._unary_op = unary_op
       self._fun      = fun
       self._argproxy = argproxy


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)

       rep.val("fun",      self._fun)
       rep.val("unary_op", self._unary_op)
       rep.val("argproxy", self._argproxy)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)

       log.val(self._fun,      other._fun)
       log.val(self._unary_op, other._unary_op)
       log.val(self._argproxy, other._argproxy)

       return bool(log)


   def __call__(self, *args, **kwargs):

       @functools.wraps(self._fun)
       def unary_fun(x): 
           return self._fun(*self._argproxy.insert(args, x), **kwargs)
            
       return self._unary_op(unary_fun, self._argproxy.extract(args))




# --- Create nary operator -------------------------------------------------- #

def nary_op(unary_op):

    @functools.wraps(unary_op)
    def wrap(fun, adx=None, *args, **kwargs):

        op = UnaryOp(unary_op, *args, **kwargs)

        return NaryOp(op, fun, argproxy(adx))

    return wrap




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Singular argument proxy (represents a single variable in args) -------- #

class ArgProxySingular(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("adx", self._adx)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._adx, other._adx)

       return bool(log)


   def insert(self, args, x):
        
       out            = list(args)
       out[self._adx] = x

       return tuple(out)


   def extract(self, args):

       return args[self._adx]




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class ArgProxyPlural(ArgProxy):

   def __init__(self, adx):

       self._adx = adx


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("adx", self._adx)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()
       log.typ(self, other)

       if bool(log):
          log.val(self._adx, other._adx)

       return bool(log)


   def insert(self, args, x):

       out = list(args)

       for i, v in zip(self._adx, x):
           out[i] = v  

       return tuple(out)


   def extract(self, args):

       return tuple(args[i] for i in self._adx)  




# --- Unary function argument proxy ----------------------------------------- # 

def argproxy(adx):

    if adx is None:
       adx = 0

    if isinstance(adx, int):
       return ArgProxySingular(adx)

    return ArgProxyPlural(adx)




