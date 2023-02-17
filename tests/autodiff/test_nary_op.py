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

       w = data.nary_op_dat(args, adx)

       assert w.nary_op(*w.args) == w.out


   @pytest.mark.parametrize("adx, proxytype", [
      [0,     "SINGULAR"],
      [1,     "SINGULAR"],
      [(0,1), "PLURAL"],
   ])  
   def test_make_nary_op(self, adx, proxytype):

       w = data.nary_op_creator_dat(adx, proxytype)

       nary_op = tdnary.make_nary_op(w.unary_op)(w.fun, w.adx)
       assert nary_op == w.nary_op


   def test_make_nary_op_001(self):

       w = data.nary_op_creator_dat(0, "SINGULAR")

       nary_op = tdnary.make_nary_op(w.unary_op)(w.fun) 
       assert nary_op == w.nary_op




























































