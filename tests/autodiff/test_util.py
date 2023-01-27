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

class TestSingularArgProxy:

   def _setup(self, args, adx, x):

       if len(args) == 0:
          return [x]

       ans = {
              0: lambda: [x,       args[1], args[2], args[3]],
              1: lambda: [args[0], x,       args[2], args[3]],
              2: lambda: [args[0], args[1], x,       args[3]],
              3: lambda: [args[0], args[1], args[2], x],
             }[adx]()

       return ans


   @pytest.mark.parametrize("args, adx, x", [
      [tpl.repeat(fake.FunReturn, 4), 0, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 1, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 2, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 3, fake.FunReturn()],
   ])  
   def test_insert(self, args, adx, x):

       ans      = self._setup(args, adx, x)
       argproxy = tdutil.SingularArgProxy(adx)

       assert argproxy.insert(args, x) == ans


   @pytest.mark.parametrize("args, adx, x", [
      [tuple(),                       0, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 0, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 1, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 2, fake.FunReturn()],
      [tpl.repeat(fake.FunReturn, 4), 3, fake.FunReturn()],
   ])  
   def test_extract(self, args, adx, x):

       args     = self._setup(args, adx, x)
       argproxy = tdutil.SingularArgProxy(adx)

       assert argproxy.extract(args) == x


   @pytest.mark.parametrize("adx", [None,0,1,2])
   def test_make_argproxy(self, adx):

       if   adx is None:
            ans = tdutil.SingularArgProxy(0)
       else:
            ans = tdutil.SingularArgProxy(adx)

       assert tdutil.make_argproxy(adx) == ans




# --- Plural argument proxy (represents an ntuple variable in args) --------- #

class TestPluralArgProxy:

   def _setup(self, args, adx, x):

       if len(args) == 0:
          return x

       ans = {
              (1,):    lambda: [args[0], x[0],    args[2], args[3]],
              (0,1):   lambda: [x[0],    x[1],    args[2], args[3]],
              (1,3):   lambda: [args[0], x[0],    args[2], x[1]],
              (0,2,3): lambda: [x[0],    args[1], x[1],    x[2]],
              (0,3):   lambda: [x[0],    args[1], args[2], x[1]],
             }[adx]()

       return ans


   @pytest.mark.parametrize("args, adx, x", [
      [tpl.repeat(fake.FunReturn, 4), (1,),    tpl.repeat(fake.FunReturn, 1)],
      [tpl.repeat(fake.FunReturn, 4), (0,1),   tpl.repeat(fake.FunReturn, 2)],
      [tpl.repeat(fake.FunReturn, 4), (1,3),   tpl.repeat(fake.FunReturn, 2)],
      [tpl.repeat(fake.FunReturn, 4), (0,2,3), tpl.repeat(fake.FunReturn, 3)],
      [tpl.repeat(fake.FunReturn, 4), (0,3),   tpl.repeat(fake.FunReturn, 2)],
   ])  
   def test_insert(self, args, adx, x):

       ans      = self._setup(args, adx, x)
       argproxy = tdutil.PluralArgProxy(adx)

       assert argproxy.insert(args, x) == ans


   @pytest.mark.parametrize("args, adx, x", [
      [tuple(),                       (0,1),   tpl.repeat(fake.FunReturn, 2)],
      [tpl.repeat(fake.FunReturn, 4), (1,),    tpl.repeat(fake.FunReturn, 1)],
      [tpl.repeat(fake.FunReturn, 4), (0,1),   tpl.repeat(fake.FunReturn, 2)],
      [tpl.repeat(fake.FunReturn, 4), (1,3),   tpl.repeat(fake.FunReturn, 2)],
      [tpl.repeat(fake.FunReturn, 4), (0,2,3), tpl.repeat(fake.FunReturn, 3)],
      [tpl.repeat(fake.FunReturn, 4), (0,3),   tpl.repeat(fake.FunReturn, 2)],
   ])  
   def test_extract(self, args, adx, x):

       args     = self._setup(args, adx, x)
       argproxy = tdutil.PluralArgProxy(adx)

       assert argproxy.extract(args) == x


   @pytest.mark.parametrize("adx", [(0,1), (1,), (0,2,3)])
   def test_make_argproxy(self, adx):

       ans = tdutil.SingularArgProxy(adx)

       assert tdutil.make_argproxy(adx) == ans






































