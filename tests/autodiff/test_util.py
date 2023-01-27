#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.util as tdutil
import tests.autodiff.fakes  as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Decorator for cacheable methods --------------------------------------- #

class TestCacheable:

   def test_cacheable(self):

       class CacheMe(fake.CacheMe):

          @tdutil.cacheable
          def compute(self):
              return super().compute()

       x   = CacheMe()
       ans = x.compute()

       assert x.compute()  == ans
       assert x.sentinel() == 1


   def test_cacheable_simple(self):

       class CacheMe(fake.CacheMe):

          @tdutil.cacheable
          def compute(self):
              return super().compute()

       x = CacheMe()
       assert x.sentinel() == 0


   def test_cacheable_advanced(self):

       class CacheMe(fake.CacheMe):

          @tdutil.cacheable
          def compute(self):
              return super().compute()

       x1   = CacheMe()
       ans1 = x1.compute()

       x2   = CacheMe()
       ans2 = x2.compute()

       assert x1.compute()  == ans1
       assert x2.compute()  == ans2
       assert x1.sentinel() == 1
       assert x2.sentinel() == 1




###############################################################################
###                                                                         ###
### Sequence data structure (quasi-immutable)                               ###
###                                                                         ###
###############################################################################


# --- Sequence -------------------------------------------------------------- #









###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Singular argument proxy (represents a single variable in args) -------- #





# --- Plural argument proxy (represents an ntuple variable in args) --------- #



# --- Unary function argument proxy ----------------------------------------- #












































