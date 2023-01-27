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


   def test_cacheable_raw(self):

       class CacheMe(fake.CacheMe):
          pass

       x   = CacheMe()
       ans = x.compute()

       assert x.compute()  != ans
       assert x.sentinel() == 2


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

class TestSequence:

   @pytest.mark.parametrize("x, xs, end", [
      [fake.FunReturn(), [],                            0],
      [fake.FunReturn(), tpl.repeat(fake.FunReturn, 2), 2],
      [fake.FunReturn(), tpl.repeat(fake.FunReturn, 2), 1],
   ])
   def test_push(self, x, xs, end):

       seq  = tdutil.Sequence(list(xs), end)
       seq1 = tdutil.Sequence([*xs, x], end+1)

       assert seq.push(x) == seq1 
       

   @pytest.mark.parametrize("xs, end", [
      [tpl.repeat(fake.FunReturn, 3), 3],
      [tpl.repeat(fake.FunReturn, 3), 2],
      [tpl.repeat(fake.FunReturn, 3), 1],
   ])
   def test_pop(self, xs, end):

       seq  = tdutil.Sequence(list(xs), end)
       seq1 = tdutil.Sequence(list(xs), end-1)

       assert seq.pop() == seq1 
       

   @pytest.mark.parametrize("x, pos, xs, end", [
      [fake.FunReturn(), 2, tpl.repeat(fake.FunReturn, 3), 3],
      [fake.FunReturn(), 1, tpl.repeat(fake.FunReturn, 3), 2],
      [fake.FunReturn(), 0, tpl.repeat(fake.FunReturn, 3), 1],
   ])
   def test_top(self, x, pos, xs, end):

       xs  = list(xs)
       xs.insert(pos, x)
       seq = tdutil.Sequence(xs, end)

       assert seq.top() == x 
              

   @pytest.mark.parametrize("xs, end, size", [
      [tpl.repeat(fake.FunReturn, 3), 3, 3],
      [tpl.repeat(fake.FunReturn, 3), 2, 2],
      [tpl.repeat(fake.FunReturn, 3), 1, 1],
      [tpl.repeat(fake.FunReturn, 3), 0, 0]
   ])
   def test_size(self, xs, end, size):

       seq = tdutil.Sequence(list(xs), end)
       assert seq.size() == size
              

   @pytest.mark.parametrize("xs, end, empty", [
      [tpl.repeat(fake.FunReturn, 3), 3, False],
      [tpl.repeat(fake.FunReturn, 3), 2, False],
      [tpl.repeat(fake.FunReturn, 3), 1, False],
      [tpl.repeat(fake.FunReturn, 3), 0, True],
   ])
   def test_empty(self, xs, end, empty):

       seq = tdutil.Sequence(list(xs), end)

       if empty:
          assert seq.empty() 
       else:
          assert not seq.empty()
       

   @pytest.mark.parametrize("x, pos, contains, xs, end", [
      [fake.FunReturn(), 2, True,  tpl.repeat(fake.FunReturn, 3), 3],
      [fake.FunReturn(), 2, False, tpl.repeat(fake.FunReturn, 3), 2],
      [fake.FunReturn(), 2, False, tpl.repeat(fake.FunReturn, 3), 1],
   ])
   def test_contains(self, x, pos, contains, xs, end):

       xs = list(xs)
       xs.insert(pos, x)
       seq = tdutil.Sequence(xs, end)

       if contains:
          assert seq.contains(x) 
       else:
          assert not seq.contains(x)
       

   @pytest.mark.parametrize("xs, end, cap", [
      [tpl.repeat(fake.FunReturn, 3), 3, 3],
      [tpl.repeat(fake.FunReturn, 3), 2, 2],
      [tpl.repeat(fake.FunReturn, 3), 1, 1],
      [tpl.repeat(fake.FunReturn, 3), 0, 0],
   ])
   def test_iterate(self, xs, end, cap):

       seq = tdutil.Sequence(list(xs), end)
       assert tuple(seq.iterate()) == xs[:cap]




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












































