#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import tadpole.util as util




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


   def __repr__(self):

       rep = util.ReprChain()

       rep.typ(self)
       rep.val("unary_op", self._unary_op)
       rep.val("fun",      self._fun)
       rep.val("argproxy", self._argproxy)

       return str(rep)


   def __eq__(self, other):

       log = util.LogicalChain()

       log.typ(self, other)
       log.val(self._unary_op, other._unary_op)
       log.val(self._fun,      other._fun)
       log.val(self._argproxy, other._argproxy)

       return bool(log)


   def __call__(self, *args, **kwargs):

       def unary_fun(x):
           return self._fun(*self._argproxy.insert(args, x), **kwargs)
            
       return self._unary_op(unary_fun, self._argproxy.extract(args))




# --- Create nary operator -------------------------------------------------- #

def nary_op(unary_op):

    def wrap(fun, adx=None):
        return NaryOp(unary_op, fun, argproxy(adx))

    return wrap




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy interface ---------------------------------------------- #

class ArgProxy(abc.ABC):

   @abc.abstractmethod
   def insert(self, args, x):
       pass

   @abc.abstractmethod
   def extract(self, args):
       pass




# --- Singular argument proxy (represents a single variable in args) -------- #

class SingularArgProxy(ArgProxy):

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
       log.val(self._adx, other._adx)

       return bool(log)


   def insert(self, args, x):
        
       out            = list(args)
       out[self._adx] = x

       return tuple(out)


   def extract(self, args):

       return args[self._adx]




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class PluralArgProxy(ArgProxy):

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
       return SingularArgProxy(adx)

    return PluralArgProxy(adx)




