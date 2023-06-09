#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential operator ------------------------------------------------- #

class TestDifferentialOp:

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_value(self, which):

       w    = data.graph_dat(which)
       prop = {
               "REVERSE": ad.PropagationReverse,
               "FORWARD": ad.PropagationForward,
              }[which](w.start, w.end)

       diffop = ad.DifferentialOp(prop)
       assert diffop.value() == w.out


   @pytest.mark.parametrize("layer", [0,1])
   def test_grad_reverse(self, layer):

       w = data.reverse_node_network_dat(layer)

       prop = ad.PropagationReverse(w.start, w.end)
       grad = w.gradmap[w.start]
       seed = w.gradmap[w.end]

       diffop = ad.DifferentialOp(prop)
       assert diffop.grad(seed) == grad


   @pytest.mark.parametrize("layer", [0,1])
   def test_grad_forward(self, layer):

       w = data.forward_node_network_dat(layer)

       prop = ad.PropagationForward(w.start, w.end)
       grad = w.gradmap[w.end] 
       seed = w.gradmap[w.start]  

       diffop = ad.DifferentialOp(prop)
       assert diffop.grad(seed) == grad




###############################################################################
###                                                                         ###
###  Gradient propagation through the AD computation graph.                 ###
###                                                                         ###
###############################################################################


# --- Forward gradient propagation ------------------------------------------ #

class TestPropagationForward:

   def test_apply(self):

       dat  = data.graph_dat("FORWARD")
       prop = ad.PropagationForward(dat.start, dat.end)

       assert prop.apply(lambda x: ag.ArgsGen(x).deshelled()[0]) == dat.out


   @pytest.mark.parametrize("layer", [0])
   def test_grads(self, layer):

       w = data.forward_node_network_dat(layer)

       prop  = ad.PropagationForward(w.start, w.end)
       seed  = w.gradmap[w.start]
       grads = ad.GradSum(seed, w.gradmap) 

       assert prop.grads(seed) == grads




# --- Reverse gradient propagation ------------------------------------------ #

class TestPropagationReverse:

   def test_apply(self):

       dat  = data.graph_dat("REVERSE")
       prop = ad.PropagationReverse(dat.start, dat.end)

       assert prop.apply(lambda x: ag.ArgsGen(x).deshelled()[0]) == dat.out


   @pytest.mark.parametrize("layer", [0])
   def test_grads(self, layer):

       w = data.reverse_node_network_dat(layer)

       prop  = ad.PropagationReverse(w.start, w.end)
       seed  = w.gradmap[w.end]
       grads = ad.GradAccum({None: w.gradmap[w.start]})

       assert prop.grads(seed) == grads




###############################################################################
###                                                                         ###
###  Function evaluation operator (builds AD computation graph)             ###
###                                                                         ###
###############################################################################


# --- Function evaluation operator ------------------------------------------ #

class TestEvalOp:

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_graph(self, which):

       dat = data.graph_dat(which)
       assert dat.evalop.graph(dat.root) == dat.graph


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_execute(self, which):

       dat        = data.graph_dat(which)
       start, end = dat.evalop.execute(dat.root)

       assert start == dat.start
       assert end   == dat.end




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Vanilla node log ------------------------------------------------------ #

class TestNodeLogVanilla:

   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_vanilla_dat_001,
      data.nodelog_vanilla_dat_002,
      data.nodelog_vanilla_dat_003,
   ])
   def test_len(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert len(w.log) == len(w.list)


   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_vanilla_dat_001,
      data.nodelog_vanilla_dat_002,
      data.nodelog_vanilla_dat_003,
   ])
   def test_bool(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert len(w.log) == bool(w.list)


   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_vanilla_dat_001,
      data.nodelog_vanilla_dat_002,
      data.nodelog_vanilla_dat_003,
   ])
   def test_iter(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert list(iter(w.log)) == w.list


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_push(self, valency):

       w = data.nodelog_vanilla_factory_dat(valency)
       x = w.log()

       assert x.push(w.node)     == w.log(w.node[0])
       assert x.push(*w.parents) == w.log(w.node[0], *w.parents[0])


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pop(self, valency):

       w = data.nodelog_vanilla_factory_dat(valency)
       x = w.log(*w.parents[0], w.node[0])
 
       assert x.pop(w.node[0]) == w.log(*w.parents[0])

       for i, parent in enumerate(w.parents[0]):
           assert x.pop(parent) == w.log(*w.parents[0][:-1-i])

       


# --- Log of effectively childless nodes ------------------------------------ # 

class TestNodeLogChildless:

   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_childless_dat_001,
      data.nodelog_childless_dat_002,
      data.nodelog_childless_dat_003,
   ])
   def test_len(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert len(w.log) == len(w.list)


   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_childless_dat_001,
      data.nodelog_childless_dat_002,
      data.nodelog_childless_dat_003,
   ])
   def test_bool(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert len(w.log) == bool(w.list)


   @pytest.mark.parametrize("valency",      [1,2,3])
   @pytest.mark.parametrize("nodelog_data", [
      data.nodelog_childless_dat_001,
      data.nodelog_childless_dat_002,
      data.nodelog_childless_dat_003,
   ])
   def test_iter(self, valency, nodelogdat):

       w = nodelogdat(valency)
       assert list(iter(w.log)) == w.list


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_push(self, valency):

       w = data.nodelog_childless_factory_dat(valency)
       x = w.log({
                  x.node: 0, **{p: 2 for p in x.parents},
                  y.node: 1, **{p: 1 for p in y.parents},
                 })

       assert x.push(w.node[0])     == w.log(w.node[0])
       assert x.push(*w.parents[0]) == w.log(w.node[0])
       assert x.push(*w.parents[0]) == w.log(w.node[0], *w.parents[0])
       assert x.push(w.node[1])     == w.log(w.node[0], *w.parents[0], w.node[1])
       assert x.push(*w.parents[1]) == w.log(w.node[0], *w.parents[0], w.node[1], *w.parents[1])


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pop(self, valency):

       w = data.nodelog_childless_factory_dat(valency)
       x = w.log(*w.parents[0], w.node[0])
 
       assert x.pop(w.node[0]) == w.log(*w.parents[0])

       for i, parent in enumerate(w.parents[0]):
           assert x.pop(parent) == w.log(*w.parents[0][:-1-i])




# --- Toposort -------------------------------------------------------------- #

class TestToposort:

   def test_childcount(self):

       w = data.reverse_node_network_dat() 
       assert ad.childcount(w.end) == w.countmap
 
       
   def test_toposort(self):

       w = data.reverse_node_network_dat() 
       assert tuple(ad.toposort(w.end)) == tuple(w.nodes)




###############################################################################
###                                                                         ###
###  Gradient summation and accumulation                                    ###
###                                                                         ###
###############################################################################


# --- Gradient summation ---------------------------------------------------- #

class TestGradSum:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       x = data.forward_node_dat(valency)

       gradmap = dict(zip(x.parents, x.seed))
       grads   = ad.GradSum(fake.Value(), gradmap)

       grads.add(x.node, x.grads)
       assert gradmap == {
                          **dict(zip(x.parents, x.seed)), 
                          x.node: sum(x.grads),
                         }


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick(self, valency):

       x = data.forward_node_dat(valency)

       gradmap = dict(zip(x.parents, x.seed))
       grads   = ad.GradSum(fake.Value(), gradmap)

       assert grads.pick(x.parents) == x.seed
       assert gradmap               == dict(zip(x.parents, x.seed))
                                       

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick_001(self, valency):

       x = data.forward_node_dat(valency)

       init_seed = fake.Value()
       gradmap   = {x.node: sum(x.grads)}
       grads     = ad.GradSum(init_seed, gradmap)

       assert grads.pick(tuple()) == (init_seed, )
       assert gradmap             == {x.node: sum(x.grads)}


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_result(self, valency):

       x = data.forward_node_dat(valency)

       gradmap = dict(zip(x.parents, x.seed))
       grads   = ad.GradSum(fake.Value(), gradmap)
       assert grads.result() == x.seed[-1]

       grads.add(x.node, x.grads)
       assert grads.result() == sum(x.grads)



# --- Gradient accumulation ------------------------------------------------- #

class TestGradAccum:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {x.node: x.seed} 
       grads   = ad.GradAccum(gradmap)

       grads.add(x.parents, x.grads)
       assert gradmap == {
                          x.node: x.seed, 
                          **dict(zip(x.parents, x.grads)),
                         }


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {**dict(zip(x.parents, x.grads)), x.node: x.seed} 
       grads   = ad.GradAccum(gradmap)

       assert grads.pick(x.node) == x.seed
       assert gradmap == {**dict(zip(x.parents, x.grads)), None: x.seed}


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_result(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {**dict(zip(x.parents, x.grads)), x.node: x.seed} 
       grads   = ad.GradAccum(gradmap)
       assert grads.result() == x.seed

       out = grads.pick(x.node)
       assert grads.result() == out 




