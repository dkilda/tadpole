#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.node     as tdnode
import tadpole.autodiff.graph    as tdgraph
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

class TestGraph:

  
   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_graph(self, graph):

       self.graph = graph


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   # --- Helpers --- #

   def _setup_forward(self, x, layer):

       nodule = self.nodule(tdnode.nodify(x), layer)
       node   = fake.ForwardNode()
       gate   = fake.ForwardGate(nodify=fake.Map({nodule: node}))

       ans = fake.ForwardNode()
       fun = fake.Fun({(node,): ans})

       return fun, gate, ans


   def _setup_reverse(self, x, layer):

       nodule = self.nodule(tdnode.nodify(x), layer)
       node   = fake.ReverseNode()
       gate   = fake.ReverseGate(nodify=fake.Map({nodule: node}))

       ans = fake.ReverseNode()
       fun = fake.Fun({(node,): ans})

       return fun, gate, ans


   # --- Tests: minilayer --- #

   def test_minilayer(self):

       assert tdgraph.minlayer() == -1


   # --- Tests: build --- #

   @pytest.mark.parametrize("x", [
                                  fake.ForwardNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])   
   def test_build_forward(self, x): 

       fun, gate, ans = self._setup_forward(x, -1)  
  
       graph = self.graph(fun, x)
       assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])   
   def test_build_reverse(self, x):      # FIXME make_node() should be more testable: 
                                         # the internally created and never returned Nodule 
                                         # makes it impossible to test using fakes alone
       fun, gate, ans = self._setup_reverse(x, -1)  

       graph = self.graph(fun, x)
       assert graph.build(gate) == ans


   # --- Tests: enter --- #

   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])  
   def test_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])  
   def test_nested_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 1) 

       with self.graph() as graph1:
          with self.graph(fun, x) as graph:
             assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])  
   def test_extra_nested_enter(self, x):

       fun, gate, ans = self._setup_reverse(x, 2) 

       with self.graph() as graph2:
          with self.graph() as graph1:
             with self.graph(fun, x) as graph:
                assert graph.build(gate) == ans


   # --- Tests: exit --- #

   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])  
   def test_nested_exit(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          with self.graph() as graph1:
             pass
          assert graph.build(gate) == ans


   @pytest.mark.parametrize("x", [
                                  fake.ReverseNode(), 
                                  fake.Point(), 
                                  fake.FunReturn(),
                                 ])  
   def test_extra_nested_exit(self, x):

       fun, gate, ans = self._setup_reverse(x, 0) 

       with self.graph(fun, x) as graph:
          with self.graph() as graph1:
             with self.graph() as graph2:
                pass
          assert graph.build(gate) == ans




###############################################################################
###                                                                         ###
###  Autodiff function decorators                                           ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Function with gate ---------------------------------------------------- #

class TestFunWithGate:

   @pytest.fixture(autouse=True)
   def request_fun_with_gate(self, fun_with_gate):

       self.fun_with_gate = fun_with_gate


   @pytest.mark.parametrize("valency", [0,1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()

       fun = self.fun_with_gate(raw_fun=fake.Fun({args: ans}))
 
       assert fun(*args) == ans


   @pytest.mark.parametrize("valency", [0,1,2,3])
   def test_gate(self, valency):

       gate  = fake.ReverseGate()
       logic = fake.ReverseLogic(gate=gate)

       fun     = fake.Fun(valency)
       vjpfuns = tpl.repeat(fake.Fun, valency) 

       tda.vjpmap.add(fun, *vjpfuns)

       assert self.fun_with_gate(diff_fun=fun).gate(logic) == gate




# --- Differentiable function decorator ------------------------------------- #

class TestDifferentiable:

   @pytest.fixture(autouse=True)
   def request_differentiable(self, differentiable):

       self.differentiable = differentiable


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()
       fun  = self.differentiable(fake.Fun({args: ans}))

       assert fun(*args) == ans




# --- Non-differentiable function decorator --------------------------------- #

class TestNonDifferentiable:

   @pytest.fixture(autouse=True)
   def request_nondifferentiable(self, nondifferentiable):

       self.nondifferentiable = nondifferentiable


   @pytest.mark.parametrize("valency", [1,2,3])
   def test_call(self, valency):

       args = tpl.repeat(fake.FunReturn, valency)
       ans  = fake.FunReturn()
       fun  = self.nondifferentiable(fake.Fun({args: ans}))

       assert fun(*args) == ans       




###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

class TestNodeTrain:

   @pytest.fixture(autouse=True)
   def request_node_train(self, node_train):

       self.node_train = node_train


   @pytest.mark.parametrize("node", [
                                     fake.ForwardNode(), 
                                     fake.ReverseNode(), 
                                     fake.Point(),
                                    ])   
   def test_with_node(self, node):

       ans   = fake.Sequence()
       nodes = fake.Sequence(push={node: ans})

       train  = self.node_train(nodes=nodes)
       train1 = self.node_train(nodes=ans)

       assert train.with_node(node) == train1


"""
   def test_with_meta(self):



   def test_concatenate(self):
"""















