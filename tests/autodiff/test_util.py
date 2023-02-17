#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.util as tdutil
import tadpole.autodiff.node as tdnode



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
          def execute(self):
              return super().execute()

       x   = CacheMe()
       ans = x.execute()

       assert x.execute()  == ans
       assert x.sentinel() == 1


   def test_cacheable_001(self):

       class CacheMe(fake.CacheMe):
          pass

       x   = CacheMe()
       ans = x.execute()

       assert x.execute()  != ans
       assert x.sentinel() == 2


   def test_cacheable_002(self):

       class CacheMe(fake.CacheMe):

          @tdutil.cacheable
          def execute(self):
              return super().execute()

       x = CacheMe()
       assert x.sentinel() == 0


   def test_cacheable_003(self):

       class CacheMe(fake.CacheMe):

          @tdutil.cacheable
          def execute(self):
              return super().execute()

       x1   = CacheMe()
       ans1 = x1.execute()

       x2   = CacheMe()
       ans2 = x2.execute()

       assert x1.execute()  == ans1
       assert x2.execute()  == ans2
       assert x1.sentinel() == 1
       assert x2.sentinel() == 1




###############################################################################
###                                                                         ###
### Automated type conversion                                               ###
###                                                                         ###
###############################################################################


# --- Conversion of type ---------------------------------------------------- #

class TestTypeConv:

   def test_typeconv(self):

       node     = tdnode.Node(fake.NodeLike(), 0, fake.GateLike())
       typeconv = tdutil.typeconv(tdnode.NodeLike, tdnode.Point)

       assert typeconv(node) is node


   def test_typeconv_001(self):

       node     = tdnode.Point(fake.Value)
       typeconv = tdutil.typeconv(tdnode.NodeLike, tdnode.Point)

       assert typeconv(node) is node


   def test_typeconv_002(self):

       source   = fake.Value()
       typeconv = tdutil.typeconv(tdnode.NodeLike, tdnode.Point)

       assert typeconv(source) == tdnode.Point(source)




# --- Conversion of type stored in an iterable ------------------------------ #

class TestIterConv:

   def test_iterconv(self):

       nodes = (
                tdnode.Node(fake.NodeLike(), 1, fake.GateLike()),  
                tdnode.Node(fake.NodeLike(), 0, fake.GateLike()),  
                tdnode.Node(fake.NodeLike(), 0, fake.GateLike()),  
               ) 

       iterconv = tdutil.iterconv(tdnode.NodeLike, tdnode.Point)
       assert iterconv(nodes) is nodes


   def test_iterconv_001(self):

       nodes = (
                tdnode.Point(fake.Value()),  
                tdnode.Point(fake.Value()),  
                tdnode.Node(fake.NodeLike(), 0, fake.GateLike()),  
               ) 

       iterconv = tdutil.iterconv(tdnode.NodeLike, tdnode.Point)
       assert iterconv(nodes) is nodes


   def test_iterconv_002(self):

       nodes = (
                tdnode.Node(fake.NodeLike(), 1, fake.GateLike()), 
                fake.Value(),  
                tdnode.Node(fake.NodeLike(), 0, fake.GateLike()),  
               ) 

       nodes1 = (
                 nodes[0],  
                 tdnode.Point(nodes[1]),
                 nodes[2],  
                ) 

       iterconv = tdutil.iterconv(tdnode.NodeLike, tdnode.Point)
       assert iterconv(nodes) == nodes1




###############################################################################
###                                                                         ###
### Sequence data structure (quasi-immutable)                               ###
###                                                                         ###
###############################################################################


# --- Sequence -------------------------------------------------------------- #

class TestSequence:

   @pytest.mark.parametrize("x, xs, end", [
      [fake.Value(), [],                            0],
      [fake.Value(), common.arepeat(fake.Value, 2), 2],
      [fake.Value(), common.arepeat(fake.Value, 2), 1],
   ])
   def test_push(self, x, xs, end):

       seq  = tdutil.Sequence(list(xs), end)
       seq1 = tdutil.Sequence([*xs, x], end+1)

       assert seq.push(x) == seq1 
       

   @pytest.mark.parametrize("xs, end", [
      [common.arepeat(fake.Value, 3), 3],
      [common.arepeat(fake.Value, 3), 2],
      [common.arepeat(fake.Value, 3), 1],
   ])
   def test_pop(self, xs, end):

       seq  = tdutil.Sequence(list(xs), end)
       seq1 = tdutil.Sequence(list(xs), end-1)

       assert seq.pop() == seq1 

              
   @pytest.mark.parametrize("xs, end, size", [
      [common.arepeat(fake.Value, 3), 3, 3],
      [common.arepeat(fake.Value, 3), 2, 2],
      [common.arepeat(fake.Value, 3), 1, 1],
      [common.arepeat(fake.Value, 3), 0, 0]
   ])
   def test_size(self, xs, end, size):

       seq = tdutil.Sequence(list(xs), end)
       assert len(seq) == size

       
   @pytest.mark.parametrize("x, pos, contains, xs, end", [
      [fake.Value(), 2, True,  common.arepeat(fake.Value, 3), 3],
      [fake.Value(), 2, False, common.arepeat(fake.Value, 3), 2],
      [fake.Value(), 2, False, common.arepeat(fake.Value, 3), 1],
   ])
   def test_contains(self, x, pos, contains, xs, end):

       xs = list(xs)
       xs.insert(pos, x)
       seq = tdutil.Sequence(xs, end)

       if contains:
          assert x in seq
       else:
          assert x not in seq 
       

   @pytest.mark.parametrize("xs, end, cap", [
      [common.arepeat(fake.Value, 3), 3, 3],
      [common.arepeat(fake.Value, 3), 2, 2],
      [common.arepeat(fake.Value, 3), 1, 1],
      [common.arepeat(fake.Value, 3), 0, 0],
   ])
   def test_iter(self, xs, end, cap):

       seq = tdutil.Sequence(list(xs), end)

       for i, x in enumerate(seq):
           assert x == xs[i]




###############################################################################
###                                                                         ###
### Customized loop iterator.                                               ###
### Defined by the first item and the next and stop functions, instead of   ###
### a range. Can be traversed in forward and reverse directions, keeps      ###
### track of the last item of the loop.                                     ###
###                                                                         ###
###############################################################################


# --- Loop iterator --------------------------------------------------------- #

class TestLoop:

   def test_iter(self):

       xs   = (1,4,7,10,13,16,19,22)
       loop = tdutil.Loop(xs[0], lambda x: x+3, lambda x: x > 15)

       for i, x in enumerate(loop):
           assert x == xs[i]


   def test_reversed(self):

       xs  = (1,4,7,10,13,16,19,22)
       rxs = (13,10,7,4,1)

       loop = tdutil.Loop(xs[0], lambda x: x+3, lambda x: x > 15)

       for i, x in enumerate(reversed(loop)):
           assert x == rxs[i]


   def test_last(self):

       xs   = (1,4,7,10,13,16,19,22)
       loop = tdutil.Loop(xs[0], lambda x: x+3, lambda x: x > 15)

       assert loop.last() == 13


   def test_first(self):

       xs   = (1,4,7,10,13,16,19,22)
       loop = tdutil.Loop(xs[0], lambda x: x+3, lambda x: x > 15)

       assert loop.first() == 1





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
      [common.arepeat(fake.Value, 4), 0, fake.Value()],
      [common.arepeat(fake.Value, 4), 1, fake.Value()],
      [common.arepeat(fake.Value, 4), 2, fake.Value()],
      [common.arepeat(fake.Value, 4), 3, fake.Value()],
   ])  
   def test_insert(self, args, adx, x):

       ans      = self._setup(args, adx, x)
       argproxy = tdutil.SingularArgProxy(adx)

       assert argproxy.insert(args, x) == ans


   @pytest.mark.parametrize("args, adx, x", [
      [tuple(),                       0, fake.Value()],
      [common.arepeat(fake.Value, 4), 0, fake.Value()],
      [common.arepeat(fake.Value, 4), 1, fake.Value()],
      [common.arepeat(fake.Value, 4), 2, fake.Value()],
      [common.arepeat(fake.Value, 4), 3, fake.Value()],
   ])  
   def test_extract(self, args, adx, x):

       args     = self._setup(args, adx, x)
       argproxy = tdutil.SingularArgProxy(adx)

       assert argproxy.extract(args) == x


   @pytest.mark.parametrize("adx", [None,0,1,2])
   def test_argproxy(self, adx):

       if   adx is None:
            ans = tdutil.SingularArgProxy(0)
       else:
            ans = tdutil.SingularArgProxy(adx)

       assert tdutil.argproxy(adx) == ans




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
      [common.arepeat(fake.Value, 4), (1,),    common.arepeat(fake.Value, 1)],
      [common.arepeat(fake.Value, 4), (0,1),   common.arepeat(fake.Value, 2)],
      [common.arepeat(fake.Value, 4), (1,3),   common.arepeat(fake.Value, 2)],
      [common.arepeat(fake.Value, 4), (0,2,3), common.arepeat(fake.Value, 3)],
      [common.arepeat(fake.Value, 4), (0,3),   common.arepeat(fake.Value, 2)],
   ])  
   def test_insert(self, args, adx, x):

       ans      = self._setup(args, adx, x)
       argproxy = tdutil.PluralArgProxy(adx)

       assert argproxy.insert(args, x) == ans


   @pytest.mark.parametrize("args, adx, x", [
      [tuple(),                       (0,1),   common.arepeat(fake.Value, 2)],
      [common.arepeat(fake.Value, 4), (1,),    common.arepeat(fake.Value, 1)],
      [common.arepeat(fake.Value, 4), (0,1),   common.arepeat(fake.Value, 2)],
      [common.arepeat(fake.Value, 4), (1,3),   common.arepeat(fake.Value, 2)],
      [common.arepeat(fake.Value, 4), (0,2,3), common.arepeat(fake.Value, 3)],
      [common.arepeat(fake.Value, 4), (0,3),   common.arepeat(fake.Value, 2)],
   ])  
   def test_extract(self, args, adx, x):

       args     = self._setup(args, adx, x)
       argproxy = tdutil.PluralArgProxy(adx)

       assert argproxy.extract(args) == x


   @pytest.mark.parametrize("adx", [(0,1), (1,), (0,2,3)])
   def test_argproxy(self, adx):

       ans = tdutil.PluralArgProxy(adx)
       assert tdutil.argproxy(adx) == ans




