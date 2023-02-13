#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tadpole.autodiff.adjoints.adjoints as tda
import tadpole.autodiff.adjoints.jvpmap   as tdjvp
import tadpole.autodiff.adjoints.vjpmap   as tdvjp

import tests.autodiff.fakes as fake
import tests.common.ntuple as tpl



###############################################################################
###                                                                         ###
###  Common code for handling adjoint functions (both JVP and VJP)          ### 
###  from the input.                                                        ###
###                                                                         ###
###############################################################################


# --- Adjoint functions ----------------------------------------------------- #

class TestAdjfun:

   @pytest.mark.parametrize("adjfun", [fake.Fun()])
   def test_make_adjfun(self, adjfun):

       assert tda.make_adjfun(adjfun) == adjfun


   def test_make_adjfun_none(self):

       adjfun = tda.make_adjfun(None)

       for n in range(4):
           out = adjfun(
                        fake.FunReturn(), 
                        fake.FunReturn(), 
                        *tpl.repeat(fake.FunReturn, n)
                       )
           assert out == 0


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_concatenate_adjfuns(self, valency):

       args    = tpl.repeat(fake.FunReturn, valency+2)
       answers = tpl.repeat(fake.FunReturn, valency)    
       adjfuns = [fake.Fun({args: ans}) for ans in answers]

       adjfun  = tda.concatenate_adjfuns(*adjfuns)
       outputs = tuple(adjfun(args[0], adx, *args[1:]) 
                                 for adx in range(valency))

       assert outputs == answers


   @pytest.mark.parametrize("valency, adxs", [
      [1, (0,)], 
      [2, (0,1)], 
      [3, (0,2)],
      [3, (1,2)],
   ])
   def test_concatenate_adjfuns_with_adxs(self, valency, adxs):

       args    = tpl.repeat(fake.FunReturn, valency+2)
       results = tpl.repeat(fake.FunReturn, valency)    
       adjfuns = [fake.Fun({args: res}) for res in results]

       adjfun  = tda.concatenate_adjfuns(*adjfuns, adxs=adxs)
       outputs = [adjfun(args[0], adx, *args[1:]) for adx in adxs]
       answers = [results[adx]                    for adx in adxs]

       assert outputs == answers
       



# --- Adjoint map ----------------------------------------------------------- #

class TestAdjMap:

   def test_get(self):

       adjfuns = tpl.repeat(fake.Fun, 2)
       fun     = fake.Fun()
       ans     = fake.Fun()  

       adjmap = tda.AdjMap(fake.Fun(fake.TrivMap(ans)))  
       adjmap.add(fun, *adjfuns)

       assert adjmap.get(fun) == ans


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency): # FIXME cannot test .get() and .add() separately

       args    = tpl.repeat(fake.FunReturn, valency+2)
       answers = tpl.repeat(fake.FunReturn, valency)    
       adjfuns = [fake.Fun({args: ans}) for ans in answers] 

       fun    = fake.Fun()
       adjmap = tda.AdjMap(lambda f: f)  
       adjmap.add(fun, *adjfuns)

       adjfun  = adjmap.get(fun)
       outputs = tuple(adjfun(args[0], adx, *args[1:]) 
                                 for adx in range(valency)) # FIXME we have to mock the effect of concatenate_adjfuns...
                                                            # it would be better if we could use fake concatenate_adjfuns
       assert outputs == answers


   def test_add_combo(self): # FIXME cannot test .get() and .add_combo() separately

       adjfun = fake.Fun()
       fun    = fake.Fun()
       ans    = fake.Fun()  

       adjmap = tda.AdjMap(fake.Fun({(adjfun,): ans})) # FIXME At the moment, .get() is the only port of AdjMap that allows us 
       adjmap.add_combo(fun, adjfun)                   # to check its content. An alternative: inject map into ctor, then use 
                                                       # __eq__ method to compare two instances of AdjMap.
       assert adjmap.get(fun) == ans




###############################################################################
###                                                                         ###
###  JVP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #

class TestNetJvpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       out   = fake.FunReturn()
       adxs  = range(valency)
       grads = tpl.repeat(fake.FunReturn, valency)
       args  = tpl.repeat(fake.FunReturn, valency)

       answers = tpl.repeat(fake.FunReturn, valency)  
       jvpfun  = fake.Fun({(g, adx, out, *args): ans 
                    for g, adx, ans in zip(grads, adxs, answers)})

       netjvpfun = tdjvp.NetJvpFun(jvpfun)
       assert tuple(netjvpfun(adxs, out, *args)(grads)) == answers   




# --- JVP map --------------------------------------------------------------- #

class TestJvpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       out     = fake.FunReturn()
       adxs    = range(valency)
       grads   = tpl.repeat(fake.FunReturn, valency)
       args    = tpl.repeat(fake.FunReturn, valency)
       answers = tpl.repeat(fake.FunReturn, valency)

       fun     = fake.Fun()
       adjfuns = [fake.Fun({(g, out, *args): ans}) 
                           for ans, g in zip(answers, grads)]

       adjmap = tdjvp.JvpMap()  
       adjmap.add(fun, *adjfuns)

       assert tuple(adjmap.get(fun)(adxs, out, *args)(grads)) == answers


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       out     = fake.FunReturn()
       adxs    = range(valency)
       grads   = tpl.repeat(fake.FunReturn, valency)
       args    = tpl.repeat(fake.FunReturn, valency)
       answers = tpl.repeat(fake.FunReturn, valency)

       fun    = fake.Fun()
       adjfun = fake.Fun({(g, adx, out, *args): ans
                          for g, adx, ans in zip(grads, adxs, answers)})

       adjmap = tdjvp.JvpMap()  
       adjmap.add_combo(fun, adjfun)

       assert tuple(adjmap.get(fun)(adxs, out, *args)(grads)) == answers




###############################################################################
###                                                                         ###
###  VJP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class TestNetVjpFun:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       grad = fake.FunReturn()
       out  = fake.FunReturn()
       adxs = range(valency)
       args = tpl.repeat(fake.FunReturn, valency)

       answers = tpl.repeat(fake.FunReturn, valency)  
       vjpfun  = fake.Fun({(grad, adx, out, *args): ans 
                    for adx, ans in zip(adxs, answers)})

       netvjpfun = tdvjp.NetVjpFun(vjpfun)
       assert tuple(netvjpfun(adxs, out, *args)(grad)) == answers  




# --- VJP map --------------------------------------------------------------- #

class TestVjpMap:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       grad    = fake.FunReturn()
       out     = fake.FunReturn()
       adxs    = range(valency)
       args    = tpl.repeat(fake.FunReturn, valency)
       answers = tpl.repeat(fake.FunReturn, valency)

       fun     = fake.Fun()
       adjfuns = [fake.Fun({(grad, out, *args): ans}) for ans in answers]

       adjmap = tdvjp.VjpMap()  
       adjmap.add(fun, *adjfuns)

       assert tuple(adjmap.get(fun)(adxs, out, *args)(grad)) == answers


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add_combo(self, valency):

       grad    = fake.FunReturn()
       out     = fake.FunReturn()
       adxs    = range(valency)
       args    = tpl.repeat(fake.FunReturn, valency)
       answers = tpl.repeat(fake.FunReturn, valency)

       fun    = fake.Fun()
       adjfun = fake.Fun({(grad, adx, out, *args): ans
                          for adx, ans in zip(adxs, answers)})

       adjmap = tdvjp.VjpMap()  
       adjmap.add_combo(fun, adjfun)

       assert tuple(adjmap.get(fun)(adxs, out, *args)(grad)) == answers








