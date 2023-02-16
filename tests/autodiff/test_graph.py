#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools

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


   # --- Test build --- #

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_build(self, which):

       dat = data.graph_dat(which, 0)

       with tdgraph.Graph(dat.root) as graph:
          assert graph.build(dat.fun, dat.x) == dat.end


   # --- Test enter --- #

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

class TestArgs:

   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_concat(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)
       assert x.args.concat() == x.concat

       
   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_pack(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)
       assert x.args.pack() == x.pack


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_eq(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)

       argsA = tdgraph.Args(x.nodes)
       argsB = tdgraph.Args(x.nodes)

       assert argsA == argsB


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_ne(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)
       y = data.args_dat(n, adxs, layers)

       argsA = tdgraph.Args(x.nodes)
       argsB = tdgraph.Args(y.nodes)

       assert argsA != argsB


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_len(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)

       assert len(x.args) == n


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_contains(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)

       for node in x.nodes:
           assert node in x.args


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_iter(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)

       for arg, node in zip(x.args, x.nodes):
           assert arg == node


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_getitem(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)

       for i, node in enumerate(x.nodes):
           assert x.args[i] == node




# --- Concatenation of nodes ------------------------------------------------ #

class TestConcatenation:

   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_attach(self, n, adxs, layers):
    
       w = data.concat_dat(n, adxs, layers)

       concat1 = w.concat_history[-1]
       concat2 = w.concat

       node, source, layer = w.attach_history[-1]

       assert concat1.attach(node, source, layer) == concat2


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),    (0,)],
      [2, (0,),    (1,)],
      [2, (1,),    (0,)],
      [2, (0,1),   (0,0)],
      [2, (0,1),   (0,2)],
      [3, (0,2),   (0,1)],
      [3, (0,2),   (2,1)],
      [3, (0,2),   (1,1)],
      [2, tuple(), tuple()],
   ])  
   def test_layer(self, n, adxs, layers):

       w = data.concat_out(n, adxs, layers)

       assert w.concat.layer() == w.layer
  

   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),    (0,)],
      [2, (0,),    (1,)],
      [2, (1,),    (0,)],
      [2, (0,1),   (0,0)],
      [2, (0,1),   (0,2)],
      [3, (0,2),   (0,1)],
      [3, (0,2),   (2,1)],
      [3, (0,2),   (1,1)],
      [2, tuple(), tuple()],
   ])  
   def test_adxs(self, n, adxs, layers):

       w = data.concat_out(n, adxs, layers)

       assert w.concat.adxs() == w.adxs


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),    (0,)],
      [2, (0,),    (1,)],
      [2, (1,),    (0,)],
      [2, (0,1),   (0,0)],
      [2, (0,1),   (0,2)],
      [3, (0,2),   (0,1)],
      [3, (0,2),   (2,1)],
      [3, (0,2),   (1,1)],
      [2, tuple(), tuple()],
   ])  
   def tests_parents(self, n, adxs, layers):

       w = data.concat_out(n, adxs, layers)

       assert w.concat.parents() == w.parents


   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),    (0,)],
      [2, (0,),    (1,)],
      [2, (1,),    (0,)],
      [2, (0,1),   (0,0)],
      [2, (0,1),   (0,2)],
      [3, (0,2),   (0,1)],
      [3, (0,2),   (2,1)],
      [3, (0,2),   (1,1)],
      [2, tuple(), tuple()],
   ])  
   def tests_deshell(self, n, adxs, layers):

       w = data.concat_out(n, adxs, layers)    

       assert w.concat.deshell() == w.deshell



###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack (of concatenated nodes) --------------------------------- #

class TestPack:

   @pytest.mark.parametrize("valency", [1,2,3])   
   @pytest.mark.parametrize("layer, result", [
      [-1, True], 
      [ 0, False], 
      [ 1, False], 
      [ 2, False],
   ])
   def test_innermost(self, valency, layer, result):

       x = data.pack_dat(valency, layer)
       assert x.pack.innermost() == result


   @pytest.mark.parametrize("valency", [1,2,3]) 
   @pytest.mark.parametrize("layer",   [0,1])
   def test_deshell(self, valency, layer):

       x = data.pack_dat(valency, layer)
       assert x.pack.deshell() == x.deshell


   @pytest.mark.parametrize("valency", [1,2,3]) 
   @pytest.mark.parametrize("layer",   [0,1])
   def test_deshelled(self, valency, layer):

       x = data.pack_dat(valency, layer)
       assert x.pack.deshelled() == x.deshelled


   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0,1])
   def test_fold_001(self, valency, layer):

       x = data.pack_dat(valency, layer)
       assert x.pack.fold(x.funwrap, x.source) == x.node


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_fold_002(self, valency):

       x = data.pack_dat(valency, -1)
       assert x.pack.fold(x.funwrap, x.source) == x.node
 



# --- Argument envelope ----------------------------------------------------- #

class TestEnvelope:

   @pytest.mark.parametrize("nargs", [1,2,3]) 
   def test_packs(self, nargs):

       x     = data.envelope_dat(nargs)
       packs = x.envelope.packs()

       for pack, xpack in itertools.zip_longest(reversed(packs), x.packs):
           assert pack == xpack


   @pytest.mark.parametrize("nargs", [1,2,3]) 
   def test_apply(self, nargs):

       x = data.envelope_dat(nargs)
       assert x.envelope.apply(x.fun) == x.value


   @pytest.mark.parametrize("nargs", [1,2,3]) 
   def test_applywrap(self, nargs):

       x = data.envelope_dat(nargs)
       assert x.envelope.applywrap(x.funwrap, x.fun) == x.nodes[-1]


   @pytest.mark.parametrize("nargs", [1,2,3]) 
   def test_apply_001(self, nargs):

       x = data.envelope_dat_001(nargs)
       assert x.envelope.apply(x.fun) == x.value


   @pytest.mark.parametrize("nargs", [1,2,3]) 
   def test_applywrap_001(self, nargs):

       x = data.envelope_dat_001(nargs)
       assert x.envelope.applywrap(x.funwrap, x.fun) == x.nodes[-1].tovalue()











