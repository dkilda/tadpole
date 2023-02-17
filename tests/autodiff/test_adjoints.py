#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.adjoints.adjoints as tda
import tadpole.autodiff.adjoints.jvpmap   as tdjvp
import tadpole.autodiff.adjoints.vjpmap   as tdvjp




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

       assert tda.make_adjfun(adjfun) == adjfun


   @pytest.mark.parametrize("nargs", [0,1,2,3])
   def test_make_adjfun_001(self, nargs):

       adjfun = tda.make_adjfun(None) 

       g    = fake.Value()
       out  = fake.Value()
       args = common.arepeat(fake.Value, nargs)

       assert adjfun(g, out, *args) == 0


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_concatenate_adjfuns(self, valency):

       g    = fake.Value()
       out  = fake.Value()
       args = common.arepeat(fake.Value, valency)

       answers = common.arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = tda.concatenate_adjfuns(*adjfuns)

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
       args = common.arepeat(fake.Value, valency)

       answers = common.arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       adjfun = tda.concatenate_adjfuns(*adjfuns, adxs=adxs)

       for adx in adxs:
           assert adjfun(g, adx, out, *args) == answers[adx]




# --- Adjoint map ----------------------------------------------------------- #

class TestAdjMap:

   def test_get(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: common.arepeat(fake.Value, len(adxs))

       adjmap = tda.AdjMap(fake.Fun(adjfun, adjfun))       
       adjmap.add_raw(fun, adjfun) 

       assert adjmap.get(fun) == adjfun


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       g    = fake.Value()
       out  = fake.Value()
       args = common.arepeat(fake.Value, valency)

       answers = common.arepeat(fake.Value, valency)
       adjfuns = [fake.Fun(ans, g, out, *args) for ans in answers]

       def fun(*args):
           return fake.Value()

       adjmap = tda.AdjMap(lambda f: f)       
       adjmap.add(fun, *adjfuns)

       for adx in range(valency):
           assert adjmap.get(fun)(g, adx, out, *args) == answers[adx]


   def test_add_combo(self):

       def fun(*args):
           return fake.Value()

       def adjfun(adxs, out, *args):
           return lambda g: common.arepeat(fake.Value, len(adxs))

       adjmap = tda.AdjMap(fake.Fun(adjfun, adjfun))       
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

       out     = fake.Value()
       adxs    = range(valency)
       grads   = common.arepeat(fake.Value, valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       jvpfuns = [fake.Fun(outputs[adx], grads[adx], adx, out, *args) 
                    for adx in adxs]

       def jvpfun(g, adx, out, *args):
           return jvpfuns[adx](g, adx, out, *args)

       netjvpfun = tdjvp.NetJvpFun(jvpfun)
       jvp       = netjvpfun(adxs, out, *args)

       assert tuple(jvp(grads)) == outputs




# --- JVP map --------------------------------------------------------------- #

class TestJvpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       out     = fake.Value()
       adxs    = range(valency)
       grads   = common.arepeat(fake.Value, valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       jvpfuns = [fake.Fun(outputs[adx], grads[adx], out, *args) 
                    for adx in adxs]

       def fun(*args):
           return fake.Value()

       jvpmap = tdjvp.JvpMap()
       jvpmap.add(fun, *jvpfuns)

       jvp = jvpmap.get(fun)(adxs, out, *args)
       assert tuple(jvp(grads)) == outputs       


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       out     = fake.Value()
       adxs    = range(valency)
       grads   = common.arepeat(fake.Value, valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       jvpfuns = [fake.Fun(outputs[adx], grads[adx], adx, out, *args) 
                    for adx in adxs]

       def jvpfun(g, adx, out, *args):
           return jvpfuns[adx](g, adx, out, *args)

       def fun(*args):
           return fake.Value()

       jvpmap = tdjvp.JvpMap()
       jvpmap.add_combo(fun, jvpfun)

       jvp = jvpmap.get(fun)(adxs, out, *args)
       assert tuple(jvp(grads)) == outputs




###############################################################################
###                                                                         ###
###  VJP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class TestNetVjpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       grad    = fake.Value()
       out     = fake.Value()
       adxs    = range(valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       vjpfuns = [fake.Fun(outputs[adx], grad, adx, out, *args) 
                    for adx in adxs]

       def vjpfun(g, adx, out, *args):
           return vjpfuns[adx](g, adx, out, *args)

       netvjpfun = tdvjp.NetVjpFun(vjpfun)
       vjp       = netvjpfun(adxs, out, *args)

       assert tuple(vjp(grad)) == outputs




# --- VJP map --------------------------------------------------------------- #

class TestVjpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       grad    = fake.Value()
       out     = fake.Value()
       adxs    = range(valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       vjpfuns = [fake.Fun(outputs[adx], grad, out, *args) 
                    for adx in adxs]

       def fun(*args):
           return fake.Value()

       vjpmap = tdvjp.VjpMap()
       vjpmap.add(fun, *vjpfuns)

       vjp = vjpmap.get(fun)(adxs, out, *args)
       assert tuple(vjp(grad)) == outputs       


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       grad    = fake.Value()
       out     = fake.Value()
       adxs    = range(valency)
       args    = common.arepeat(fake.Value, valency)
       outputs = common.arepeat(fake.Value, valency)

       vjpfuns = [fake.Fun(outputs[adx], grad, adx, out, *args) 
                    for adx in adxs]

       def vjpfun(g, adx, out, *args):
           return vjpfuns[adx](g, adx, out, *args)

       def fun(*args):
           return fake.Value()

       vjpmap = tdvjp.VjpMap()
       vjpmap.add_combo(fun, vjpfun)

       vjp = vjpmap.get(fun)(adxs, out, *args)
       assert tuple(vjp(grad)) == outputs




