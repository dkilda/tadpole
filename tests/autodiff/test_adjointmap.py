#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util                as util
import tadpole.autodiff.types      as at
import tadpole.autodiff.adjointmap as adj




###############################################################################
###                                                                         ###
###  Setting up adjoint functions                                           ###
###                                                                         ###
###############################################################################


# --- Setting up adjoint functions ------------------------------------------ #

class TestMake:

   @pytest.mark.parametrize("which", ["vjp", "jvp"])
   @pytest.mark.parametrize("adx",   [0,1,2])
   def test_make_adjfun(self, which, adx):

       make_adjfun = {
                      "vjp": adj.make_vjpfun,
                      "jvp": adj.make_jvpfun,
                     }[which]

       def adjfun(*args):
           return fake.Value()

       assert make_adjfun(adjfun, adx=adx) == adjfun


   @pytest.mark.parametrize("which", ["vjp", "jvp"])
   @pytest.mark.parametrize("nargs", [1,2,3])
   def test_make_adjfun_identity(self, which, nargs):

       adjfun = {
                 "vjp": adj.make_vjpfun,
                 "jvp": adj.make_jvpfun,
                }[which]("identity") 

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, nargs)

       assert adjfun(g, out, *args) == g


   @pytest.mark.parametrize("nargs, adx", [
      [1, 0],
      [1, 0],
      [2, 0],
      [2, 1],
      [3, 0],
      [3, 1],
      [3, 2],
   ])
   def test_make_vjpfun_null(self, nargs, adx):

       adjfun = adj.make_vjpfun("null", adx=adx) 

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, nargs)

       assert adjfun(g, out, *args) == args[adx].tonull()


   @pytest.mark.parametrize("nargs, adx", [
      [1, 0],
      [1, 0],
      [2, 0],
      [2, 1],
      [3, 0],
      [3, 1],
      [3, 2],
   ])
   def test_make_jvpfun_null(self, nargs, adx):

       adjfun = adj.make_jvpfun("null", adx=adx) 

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, nargs)

       assert adjfun(g, out, *args) == out.tonull()


   @pytest.mark.parametrize("nargs, adx", [
      [1, 0],
      [2, 0],
      [2, 1],
      [3, 0],
      [3, 1],
      [3, 2],
   ])
   def test_make_jvpfun_linear(self, nargs, adx):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, nargs)

       gargs = {
                0: (            g, *args[1:]),
                1: (*args[0],   g, *args[2:]),
                2: (*args[0:2], g, *args[3:]),
               }[adx]
       gout = fake.Value()
       gfun = fake.Fun(gout, *gargs)

       adjfun = adj.make_jvpfun("linear", gfun, adx) 

       assert adjfun(g, out, *args) == gout




###############################################################################
###                                                                         ###
###  Constructing net (concatenated) adjoint functions                      ###
###                                                                         ###
###############################################################################


# --- Concatenate adjoint functions ----------------------------------------- #

class TestConcat:

   @pytest.mark.parametrize("which",   ["vjp", "jvp"])
   @pytest.mark.parametrize("valency", [1,2,3])
   def test_concat_adjfuns(self, which, valency):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, valency)

       answers = arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = {
                 "vjp": adj.concat_vjpfuns, 
                 "jvp": adj.concat_jvpfuns,
                }[which](fake.Fun(None), *adjfuns)

       for adx in range(valency):
           assert adjfun(g, adx, out, *args) == answers[adx]


   @pytest.mark.parametrize("which", ["vjp", "jvp"])
   @pytest.mark.parametrize("valency, adxs", [
      [1, (0,)], 
      [2, (0,1)], 
      [3, (0,2)],
      [3, (1,2)],
   ])
   def test_concat_adjfuns_001(self, which, valency, adxs):

       g    = fake.Value()
       out  = fake.Value()
       args = arepeat(fake.Value, valency)

       answers = arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = {
                 "vjp": adj.concat_vjpfuns, 
                 "jvp": adj.concat_jvpfuns,
                }[which](fake.Fun(None), *adjfuns, adxs=adxs)

       for adx in adxs:
           assert adjfun(g, adx, out, *args) == answers[adx]




# --- Net (concatenated) VJP function --------------------------------------- #

class TestNetVjpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       w = data.vjpmap_dat(valency)

       vjp = w.netvjpfun(w.adxs, w.out, *w.args)
       assert tuple(vjp(w.grad)) == w.outputs




# --- Net (concatenated) JVP function --------------------------------------- #

class TestNetJvpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       w = data.jvpmap_dat(valency)

       jvp = w.netjvpfun(w.adxs, w.out, *w.args)
       assert tuple(jvp(w.grads)) == w.outputs




###############################################################################
###                                                                         ###
###  Adjoint maps                                                           ###
###                                                                         ###
###############################################################################


# --- Generic adjoint map --------------------------------------------------- #

class TestAdjointMap:

   def test_get(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: arepeat(fake.Value, len(adxs))

       adjmap = adj.AdjointMap(adj.concat_vjpfuns, fake.Fun(adjfun, adjfun))       
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

       adjmap = adj.AdjointMap(adj.concat_vjpfuns, lambda f: f)       
       adjmap.add(fun, *adjfuns)

       for adx in range(valency):
           assert adjmap.get(fun)(g, adx, out, *args) == answers[adx]


   def test_add_combo(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: arepeat(fake.Value, len(adxs))

       adjmap = adj.AdjointMap(adj.concat_vjpfuns, fake.Fun(adjfun, adjfun))       
       adjmap.add_combo(fun, adjfun) 

       assert adjmap.get(fun) == adjfun




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
    

   @pytest.mark.parametrize("valency, adx", [
      [1, 0],
      [2, 0],
      [2, 1],
      [3, 0],
      [3, 1],
      [3, 2],
   ])
   def test_add_combo_linear(self, valency, adx):

       w = data.jvpmap_dat(valency)

       gargs = {
                1: lambda: {
                            0: (w.grads[0], *w.args[1:])
                           },
                2: lambda: {                   
                            0: (              w.grads[0], *w.args[1:]),
                            1: (*w.args[0],   w.grads[1], *w.args[2:]),
                           },
                3: lambda: {
                            0: (              w.grads[0], *w.args[1:]),
                            1: (*w.args[0],   w.grads[1], *w.args[2:]),
                            2: (*w.args[0:2], w.grads[2], *w.args[3:]),
                           },
               }[valency]
 
       fmap = fake.Map({gargs()[adx]: w.outputs[adx] for adx in w.adxs})
       fun  = fake.Fun(fmap)

       jvpmap = adj.JvpMap()
       jvpmap.add_combo(fun, "linear")  

       jvp = jvpmap.get(fun)(w.adxs, w.out, *w.args)
       assert tuple(jvp(w.grads)) == w.outputs
              



