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
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class TestGraph:

   # --- Test minilayer --- #

   def test_minilayer(self):

       assert tdgraph.minlayer() == -1


   # --- Test enter and build --- #

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_enter(self, which):

       dat = data.graph_dat(which, 0)

       with tdgraph.Graph(dat.root) as graph:
          assert graph.build(dat.fun, dat.x) == dat.end


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_single_nested_enter(self, which):

       dat = data.graph_dat(which, 1)

       with tdgraph.Graph(fake.GateLike()) as graph0:
          with tdgraph.Graph(dat.root) as graph1:
             assert graph1.build(dat.fun, dat.x) == dat.end


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_double_nested_enter(self, which):

       dat = data.graph_dat(which, 2)

       with tdgraph.Graph(fake.GateLike()) as graph0:
          with tdgraph.Graph(fake.GateLike()) as graph1:
             with tdgraph.Graph(dat.root) as graph2:
                assert graph2.build(dat.fun, dat.x) == dat.end


   # --- Test exit and build --- #

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_single_nested_exit(self, which):

       dat = data.graph_dat(which, 0)

       with tdgraph.Graph(dat.root) as graph0:
          with tdgraph.Graph(fake.GateLike()) as graph1:
             pass
          assert graph0.build(dat.fun, dat.x) == dat.end


   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_double_nested_exit(self, which):

       dat = data.graph_dat(which, 0)

       with tdgraph.Graph(dat.root) as graph0:
          with tdgraph.Graph(fake.GateLike()) as graph1:
             with tdgraph.Graph(fake.GateLike()) as graph2:
                pass
          assert graph0.build(dat.fun, dat.x) == dat.end




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable function wrap ------------------------------------------ #

class TestDifferentiable:

   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])   
   def test_call(self, n, adxs, layers):

       args_dat = data.args_dat(n, adxs, layers)
       x        = data.differentiable_funwrap_dat(args_dat.nodes)

       assert x.funwrap(*x.args) == x.out
 



# --- Non-differentiable function wrap -------------------------------------- #

class TestNonDifferentiable:

   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])   
   def test_call(self, n, adxs, layers):

       args_dat = data.args_dat(n, adxs, layers)
       x        = data.nondifferentiable_funwrap_dat(args_dat.nodes)

       assert x.funwrap(*x.args) == x.out




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation                             ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

#class Test:




















































