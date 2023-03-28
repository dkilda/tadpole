#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util          as util
import tadpole.autodiff.nary as nary




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

       w = data.nary_op_dat(args, adx)

       assert w.nary_op(*w.args) == w.out


   @pytest.mark.parametrize("adx, proxytype", [
      [0,     "SINGULAR"],
      [1,     "SINGULAR"],
      [(0,1), "PLURAL"],
   ])  
   def test_nary_op(self, adx, proxytype):

       w = data.nary_op_creator_dat(adx, proxytype)

       nary_op = nary.nary_op(w.unary_op)(w.fun, w.adx)
       assert nary_op == w.nary_op


   def test_nary_op_001(self):

       w = data.nary_op_creator_dat(0, "SINGULAR")

       nary_op = nary.nary_op(w.unary_op)(w.fun) 
       assert nary_op == w.nary_op




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Singular argument proxy (represents a single variable in args) -------- #

class TestSingularArgProxy:

   @pytest.mark.parametrize("adx", [0,1,2,3])
   def test_insert(self, adx):

       w = data.singular_argproxy_dat(adx)

       assert w.argproxy.insert(w.args, w.x) == w.args1


   @pytest.mark.parametrize("adx", [0,1,2,3])
   def test_extract(self, adx):

       w = data.singular_argproxy_dat(adx)
  
       assert w.argproxy.extract(w.args1) == w.x


   def test_extract_001(self):

       w = data.singular_argproxy_dat_001()
  
       assert w.argproxy.extract(w.args1) == w.x


   @pytest.mark.parametrize("adx", [None, 0, 1, 2])
   def test_argproxy(self, adx):

       if   adx is None:
            ans = nary.SingularArgProxy(0)
       else:
            ans = nary.SingularArgProxy(adx)

       assert nary.argproxy(adx) == ans       




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class TestPluralArgProxy:

   @pytest.mark.parametrize("adx", [
      (1,), (0,1), (1,3), (0,2,3), (0,3)
   ])
   def test_insert(self, adx):

       w = data.plural_argproxy_dat(adx)

       assert w.argproxy.insert(w.args, w.x) == w.args1


   @pytest.mark.parametrize("adx", [
      (1,), (0,1), (1,3), (0,2,3), (0,3)
   ])
   def test_extract(self, adx):

       w = data.plural_argproxy_dat(adx)
  
       assert w.argproxy.extract(w.args1) == w.x


   def test_extract_001(self):

       w = data.plural_argproxy_dat_001()
  
       assert w.argproxy.extract(w.args1) == w.x


   @pytest.mark.parametrize("adx", [
      (0,1), (1,), (0,2,3)
   ])
   def test_argproxy(self, adx):

       ans = nary.PluralArgProxy(adx)
       assert nary.argproxy(adx) == ans




