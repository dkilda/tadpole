#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from tests.common import arepeat, arange, amap

import tests.util.fakes as fake
import tests.util.data  as data

import tadpole.util as util




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

       w = data.loop_dat()

       for i, x in enumerate(w.loop):
           assert x == w.xs[i]


   def test_reversed(self):

       w = data.loop_dat()

       for i, x in enumerate(reversed(w.loop)):
           assert x == w.reversed_xs[i]


   def test_last(self):

       w = data.loop_dat()
       assert w.loop.last() == w.last




###############################################################################
###                                                                         ###
### Sequence data structure (quasi-immutable)                               ###
###                                                                         ###
###############################################################################


# --- Sequence -------------------------------------------------------------- #

class TestSequence:

   @pytest.mark.parametrize("items, xs", [
      [tuple(),                arepeat(fake.Value, 1)],   
      [arepeat(fake.Value, 3), arepeat(fake.Value, 1)],
      [arepeat(fake.Value, 3), arepeat(fake.Value, 2)],
      [arepeat(fake.Value, 3), arepeat(fake.Value, 3)],
   ])
   def test_push(self, items, xs):

       seq  = util.Sequence(items)
       seq1 = util.Sequence([*items, *xs])

       for x in xs:
           seq = seq.push(x)

       assert seq == seq1 
       

   @pytest.mark.parametrize("items, n", [
      [arepeat(fake.Value, 3), 1],
      [arepeat(fake.Value, 3), 2],
      [arepeat(fake.Value, 3), 3],
   ])
   def test_pop(self, items, n):

       seq  = util.Sequence(items)
       seq1 = util.Sequence(items[:-n])

       for _ in range(n):
           seq = seq.pop()

       assert seq == seq1 

              
   @pytest.mark.parametrize("items, size", [
      [tuple(),                0],
      [arepeat(fake.Value, 1), 1],
      [arepeat(fake.Value, 2), 2],
      [arepeat(fake.Value, 3), 3],
   ])
   def test_size(self, items, size):

       assert len(util.Sequence(items)) == size


   @pytest.mark.parametrize("items, x", [
      [tuple(),                fake.Value()],
      [arepeat(fake.Value, 1), fake.Value()],
      [arepeat(fake.Value, 2), fake.Value()],
      [arepeat(fake.Value, 3), fake.Value()],
   ])
   def test_contains(self, items, x):

       seq = util.Sequence(items)

       for item in items:
           assert item in seq

       assert not x in seq


   @pytest.mark.parametrize("items, n, npush, npop", [
      [arepeat(fake.Value, 1), 0, 1, 0],   
      [arepeat(fake.Value, 4), 3, 1, 0],
      [arepeat(fake.Value, 5), 3, 2, 0],
      [arepeat(fake.Value, 6), 3, 3, 1],
   ])
   def test_iter(self, items, n, npush, npop):

       seq = util.Sequence(items[:n])

       for i in range(n, npush): 
           seq = seq.push(items[i])

       for _ in range(npop):
           seq = seq.pop()

       ans = items[ : (n + npush - npop)]

       for i, x in enumerate(seq):
           assert x == ans[i]









