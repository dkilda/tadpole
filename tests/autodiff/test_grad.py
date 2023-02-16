#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.util  as tdutil
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad

import tadpole.autodiff.adjoints as tda




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################
"""

# --- Forward differential operator ----------------------------------------- #

class TestForwardDifferentialOp:

   @pytest.mark.parametrize("layer", [0])
   def test_graphop(self, layer):

       dat = data.diffop_dat("FORWARD", layer)
       assert dat.diffop.graphop() == dat.graphop


   @pytest.mark.parametrize("layer", [0])
   def test_end(self, layer):

       dat = data.diffop_dat("FORWARD", layer)
       assert dat.diffop.end() == dat.end


   @pytest.mark.parametrize("layer", [0])
   def test_evaluate(self, layer):

       dat = data.diffop_dat("FORWARD", layer)
       assert dat.diffop.evaluate() == dat.out


   @pytest.mark.parametrize("layer", [0])
   def test_accum(self, layer):

       dat = data.diffop_dat("FORWARD", layer)

       seed  = fake.Value()
       accum = tdgrad.GradSum(seed)

       assert dat.diffop.accum(seed) == accum


   @pytest.mark.parametrize("layer", [0])
   def test_grad(self, layer):

       network = data.forward_node_network_dat(layer=None)

       start = network.leaves[0]
       end   = network.end

       seed = network.gradmap[start]
       grad = network.gradmap[end]

       dat = data.diffop_dat("FORWARD", layer, start=start, end=end)
       assert dat.diffop.grad(seed) == grad

"""



 




# --- Reverse differential operator ----------------------------------------- #







# --- Graph operator -------------------------------------------------------- #

class TestGraphOp:

   def test_end(self):

       dat = data.diffop_dat("FORWARD", 0)
       assert dat.graphop.end() == dat.end




















































