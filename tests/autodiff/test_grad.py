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






###############################################################################
###                                                                         ###
###  Gradient propagation through the AD computation graph.                 ###
###                                                                         ###
###############################################################################


# --- Forward gradient propagation ------------------------------------------ #

class TestForwardPropagation:

   def test_graphop(self):

       dat  = data.graph_dat("FORWARD")
       prop = tdgrad.ForwardPropagation()

       assert prop.graphop(dat.fun, dat.x) == dat.graphop


   @pytest.mark.parametrize("layer", [0])
   def test_accum(self, layer):

       network = data.forward_node_network_dat(layer)
       prop    = tdgrad.ForwardPropagation()

       end   = network.end
       start = network.leaves[0]
       seed  = network.gradmap[start]

       grads = tdgrad.GradSum(seed, network.gradmap) 

       assert prop.accum(end, seed) == grads




# --- Reverse gradient propagation ------------------------------------------ #

class TestReversePropagation:

   def test_graphop(self):

       dat  = data.graph_dat("REVERSE")
       prop = tdgrad.ReversePropagation()

       assert prop.graphop(dat.fun, dat.x) == dat.graphop


   @pytest.mark.parametrize("layer", [0])
   def test_accum(self, layer):

       network = data.reverse_node_network_dat(layer)
       prop    = tdgrad.ReversePropagation()

       end   = network.end
       start = network.leaves[0]
       seed  = network.gradmap[network.end]

       grads = tdgrad.GradAccum({None: network.gradmap[start]})

       assert prop.accum(end, seed) == grads




###############################################################################
###                                                                         ###
###  Computation graph operator.                                            ###
###                                                                         ###
###############################################################################


# --- Graph operator -------------------------------------------------------- #

class TestGraphOp:

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_graph(self, which):

       dat = data.graph_dat(which)
       assert dat.graphop.graph() == dat.graph


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_end(self, which):

       dat = data.graph_dat(which)
       assert dat.graphop.end() == dat.end


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_evaluate(self, which):

       dat = data.graph_dat(which)
       assert dat.graphop.evaluate() == dat.out



















































