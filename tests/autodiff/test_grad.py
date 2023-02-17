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


# --- Differential operator ------------------------------------------------- #

class TestDifferentialOp:

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_graphop(self, which):

       w = data.diffop_dat(which)
       assert w.diffop.graphop() == w.graphop


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_end(self, which):

       w = data.diffop_dat(which)
       assert w.diffop.end() == w.end


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_evaluate(self, which):

       w = data.diffop_dat(which)
       assert w.diffop.evaluate() == w.out




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




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Child count ----------------------------------------------------------- #

class TestChildCount:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_record(self, valency):

       w = data.childcount_dat(valency)

       assert w.count.record(w.node,  w.parents)  == w.count1
       assert w.count.record(w.node1, w.parents1) == w.count2


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_collect(self, valency):

       w = data.childcount_dat(valency)

       assert w.count.collect(w.node) == w.parents       
       assert w.parentmap             == w.parentmap1

       assert w.count.collect(w.node1) == w.parents1
       assert w.parentmap              == w.parentmap2


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_increase(self, valency):

       w = data.childcount_dat(valency)

       countmap = {}
       count    = tdgrad.ChildCount(w.parentmap2, countmap)

       assert count.increase(w.node) == w.parents
       assert countmap               == {w.node: 1} 

       assert count.increase(w.node1) == w.parents1         
       assert countmap                == {w.node: 1, w.node1: 1} 

       assert count.increase(w.node) == tuple() 
       assert countmap               == {w.node: 2, w.node1: 1} 


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_decrease(self, valency):

       w = data.childcount_dat(valency)

       countmap  = {
                    **{p: 2 for p in w.parents},
                    **{p: 1 for p in w.parents1},
                   }
       countmap1 = {
                    **{p: 1 for p in w.parents},
                    **{p: 1 for p in w.parents1},
                   }
       countmap2 = {
                    **{p: 1 for p in w.parents},
                    **{p: 0 for p in w.parents1},
                   }
       countmap3 = {
                    **{p: 0 for p in w.parents},
                    **{p: 0 for p in w.parents1},
                   }

       count = tdgrad.ChildCount(w.parentmap2, countmap)

       assert count.decrease(w.node) == tuple()
       assert countmap == countmap1

       assert count.decrease(w.node1) == tuple(w.parents1)
       assert countmap == countmap2 

       assert count.decrease(w.node1) == tuple()
       assert countmap == countmap2 

       assert count.decrease(w.node) == tuple(w.parents)
       assert countmap == countmap3 




# --- Traversal ------------------------------------------------------------- #

class TestTraversal:

   def test_sweep(self):

       dat       = data.reverse_node_network_dat()
       traversal = tdgrad.Traversal(dat.end)

       nodes = list(dat.nodes)

       def step(x):
           try:
              return (nodes[nodes.index(x) + 1], )    
           except IndexError:
              return tuple()

       for i, node in enumerate(traversal.sweep(step)):
           assert node == nodes[i]
       

   def test_apply(self):

       dat       = data.reverse_node_network_dat()
       traversal = tdgrad.Traversal(dat.end)

       parentmap = {}
       count     = tdgrad.ChildCount(parentmap)

       traversal.apply(count.collect)
       assert parentmap == dat.parentmap




# --- Topological sort ------------------------------------------------------ #

class TestTopoSort:

   def test_traverse(self):

       dat = data.reverse_node_network_dat() 

       parentmap = {}
       countmap  = {}
       count     = tdgrad.ChildCount(parentmap, countmap) 
       traversal = tdgrad.Traversal(dat.end) 

       toposort = tdgrad.TopoSort(traversal, count)
       
       for i, node in enumerate(toposort.traverse()):
           assert node == dat.nodes[i]
           

   def test_toposort(self):

       dat = data.reverse_node_network_dat()

       for i, node in enumerate(tdgrad.toposort(dat.end)):
           assert node == dat.nodes[i]




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
       grads   = tdgrad.GradSum(fake.Value(), gradmap)

       grads.add(x.node, x.grads)
       assert gradmap == {
                          **dict(zip(x.parents, x.seed)), 
                          x.node: sum(x.grads),
                         }


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick(self, valency):

       x = data.forward_node_dat(valency)

       gradmap = dict(zip(x.parents, x.seed))
       grads   = tdgrad.GradSum(fake.Value(), gradmap)

       assert grads.pick(x.parents) == x.seed
       assert gradmap               == dict(zip(x.parents, x.seed))
                                       

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick_001(self, valency):

       x = data.forward_node_dat(valency)

       init_seed = fake.Value()
       gradmap   = {x.node: sum(x.grads)}
       grads     = tdgrad.GradSum(init_seed, gradmap)

       assert grads.pick(tuple()) == (init_seed, )
       assert gradmap             == {x.node: sum(x.grads)}


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_result(self, valency):

       x = data.forward_node_dat(valency)

       gradmap = dict(zip(x.parents, x.seed))
       grads   = tdgrad.GradSum(fake.Value(), gradmap)
       assert grads.result() == x.seed[-1]

       grads.add(x.node, x.grads)
       assert grads.result() == sum(x.grads)



# --- Gradient accumulation ------------------------------------------------- #

class TestGradAccum:

   @pytest.mark.parametrize("valency", [1,2,3])
   def test_add(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {x.node: x.seed} 
       grads   = tdgrad.GradAccum(gradmap)

       grads.add(x.parents, x.grads)
       assert gradmap == {
                          x.node: x.seed, 
                          **dict(zip(x.parents, x.grads)),
                         }


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_pick(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {**dict(zip(x.parents, x.grads)), x.node: x.seed} 
       grads   = tdgrad.GradAccum(gradmap)

       assert grads.pick(x.node) == x.seed
       assert gradmap == {**dict(zip(x.parents, x.grads)), None: x.seed}


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_result(self, valency):

       x = data.reverse_node_dat(valency)

       gradmap = {**dict(zip(x.parents, x.grads)), x.node: x.seed} 
       grads   = tdgrad.GradAccum(gradmap)
       assert grads.result() == x.seed

       out = grads.pick(x.node)
       assert grads.result() == out 




