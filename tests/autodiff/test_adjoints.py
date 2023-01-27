#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints.adjoints as tda
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






###############################################################################
###                                                                         ###
###  JVP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #




# --- JVP map --------------------------------------------------------------- #







###############################################################################
###                                                                         ###
###  VJP map                                                                ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #





# --- VJP map --------------------------------------------------------------- #









