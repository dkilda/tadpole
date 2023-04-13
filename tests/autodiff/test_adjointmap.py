#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util                as util
import tadpole.autodiff.adjointmap as adj




###############################################################################
###                                                                         ###
###  Common code for handling adjoint functions (both JVP and VJP)          ### 
###  from the input.                                                        ###
###                                                                         ###
###############################################################################


# --- Adjoint functions ----------------------------------------------------- #

class TestAdjfun:

   def test_make_adjfun(self):

       def adjfun(*args):
           return fake.Value()

       assert adj.make_adjfun(adjfun) == adjfun


   @pytest.mark.parametrize("nargs", [0,1,2,3])
   def test_make_adjfun_001(self, nargs):

       adjfun = adj.make_adjfun(None) 

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, nargs)

       assert adjfun(g, out, *args) == 0


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_concatenate_adjfuns(self, valency):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, valency)

       answers = arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = adj.concatenate_adjfuns(fake.Fun(None), *adjfuns)

       for adx in range(valency):
           assert adjfun(g, adx, out, *args) == answers[adx]


   @pytest.mark.parametrize("valency, adxs", [
      [1, (0,)], 
      [2, (0,1)], 
      [3, (0,2)],
      [3, (1,2)],
   ])
   def test_concatenate_adjfuns_001(self, valency, adxs):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, valency)

       answers = arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = adj.concatenate_adjfuns(fake.Fun(None), *adjfuns, adxs=adxs)

       for adx in adxs:
           assert adjfun(g, adx, out, *args) == answers[adx]




# --- Adjoint map ----------------------------------------------------------- #

class TestAdjointMap:

   def test_get(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: arepeat(fake.Value, len(adxs))

       adjmap = adj.AdjointMap(fake.Fun(adjfun, adjfun))       
       adjmap.add_raw(fun, adjfun) 

       assert adjmap.get(fun) == adjfun


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, valency)

       answers = arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       def fun(*args):
           return fake.Value()

       adjmap = adj.AdjointMap(lambda f: f)       
       adjmap.add(fun, *adjfuns)

       for adx in range(valency):
           assert adjmap.get(fun)(g, adx, out, *args) == answers[adx]


   def test_add_combo(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: arepeat(fake.Value, len(adxs))

       adjmap = adj.AdjointMap(fake.Fun(adjfun, adjfun))       
       adjmap.add_combo(fun, adjfun) 

       assert adjmap.get(fun) == adjfun
              



###############################################################################
###                                                                         ###
###  JVP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #

class TestNetJvpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       w = data.jvpmap_dat(valency)

       jvp = w.netjvpfun(w.adxs, w.out, *w.args)
       assert tuple(jvp(w.grads)) == w.outputs




# --- JVP map --------------------------------------------------------------- #

class TestJvpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       w = data.jvpmap_dat(valency)

       w.jvpmap.add(w.fun, *w.jvpfuns)

       jvp = w.jvpmap.get(w.fun)(w.adxs, w.out, *w.args)
       assert tuple(jvp(w.grads)) == w.outputs 
    

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       w = data.jvpmap_dat(valency)

       w.jvpmap.add_combo(w.fun, w.concat_jvpfun)

       jvp = w.jvpmap.get(w.fun)(w.adxs, w.out, *w.args)
       assert tuple(jvp(w.grads)) == w.outputs  




###############################################################################
###                                                                         ###
###  VJP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class TestNetVjpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       w = data.vjpmap_dat(valency)

       vjp = w.netvjpfun(w.adxs, w.out, *w.args)
       assert tuple(vjp(w.grad)) == w.outputs




# --- VJP map --------------------------------------------------------------- #

class TestVjpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       w = data.vjpmap_dat(valency)

       w.vjpmap.add(w.fun, *w.vjpfuns)

       vjp = w.vjpmap.get(w.fun)(w.adxs, w.out, *w.args)
       assert tuple(vjp(w.grad)) == w.outputs 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       w = data.vjpmap_dat(valency)

       w.vjpmap.add_combo(w.fun, w.concat_vjpfun)

       vjp = w.vjpmap.get(w.fun)(w.adxs, w.out, *w.args)
       assert tuple(vjp(w.grad)) == w.outputs 





