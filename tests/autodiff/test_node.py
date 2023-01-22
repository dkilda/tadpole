#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.node     as tdnode
import tests.autodiff.fakes      as fake

from tests.autodiff.common import value_eq




###############################################################################
###                                                                         ###
###  Boilerplate code for testing forward and reverse objects.              ###
###                                                                         ###
###############################################################################


# --- Boilerplate for testing forward objects ------------------------------- #

class TestForward:

   @pytest.fixture(autouse=True)
   def request_randn(self, randn):

       self.randn = randn


   @pytest.fixture(autouse=True)
   def request_args(self, jvpfun_args):

       self.args = jvpfun_args


   @pytest.fixture(autouse=True)
   def request_logic(self, forward_logic):

       self.logic = forward_logic


   @pytest.fixture(autouse=True)
   def request_gate(self, forward_gate):

       self.gate = forward_gate


   @pytest.fixture(autouse=True)
   def request_node(self, forward_node):

       self.node = forward_node




# --- Boilerplate for testing reverse objects ------------------------------- #

class TestReverse:

   @pytest.fixture(autouse=True)
   def request_randn(self, randn):

       self.randn = randn


   @pytest.fixture(autouse=True)
   def request_args(self, vjpfun_args):

       self.args = vjpfun_args


   @pytest.fixture(autouse=True)
   def request_logic(self, reverse_logic):

       self.logic = reverse_logic


   @pytest.fixture(autouse=True)
   def request_gate(self, reverse_gate):

       self.gate = reverse_gate


   @pytest.fixture(autouse=True)
   def request_node(self, reverse_node):

       self.node = reverse_node




###############################################################################
###                                                                         ###
###  Logic of forward and reverse propagation, creates logic gates.         ###
###                                                                         ###
###############################################################################


# --- Forward logic --------------------------------------------------------- #

class TestForwardLogic(TestForward):

   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_gate(self, valency):

       fun     = fake.Fun()
       jvpfuns = [fake.Fun()]*valency

       tda.jvpmap.add(fun, *jvpfuns) # FIXME having to do this is an argument for keeping vjp/jvp 
                                     #       info in Fun classes instead of using global adjoint maps!
       parents = tuple([fake.ForwardNode()]*valency)
       logic   = self.logic(parents)
       ans     = self.gate(parents, fun, self.randn())

       assert logic.gate(fun) == ans


   @pytest.mark.parametrize("valency", [1, 2, 3])
   def test_make_logic(self, valency):

       ans = fake.ForwardLogic()

       adxs, out, args = self.args(valency)
       parents         = tuple([fake.ForwardNode(logic=ans)]*valency)

       assert tdnode.make_logic(parents, adxs, out, args) == ans




# --- Reverse logic --------------------------------------------------------- #

class TestReverseLogic(TestReverse):

   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_gate(self, valency):

       fun     = fake.Fun(valency)
       vjpfuns = [fake.Fun()]*valency

       tda.vjpmap.add(fun, *vjpfuns) # FIXME having to do this is an argument for keeping vjp/jvp 
                                     #       info in Fun classes instead of using global adjoint maps!

       parents = tuple([fake.ReverseNode()]*valency)
       logic   = self.logic(parents)
       ans     = self.gate(parents, fun, tda.vjpmap.get(fun)) # FIXME in the future we should also test vjp/jvp equality!

       assert logic.gate(fun) == ans


   @pytest.mark.parametrize("valency", [1, 2, 3])
   def test_make_logic(self, valency):

       ans = fake.ReverseLogic()

       adxs, out, args = self.args(valency)
       parents         = tuple([fake.ReverseNode(logic=ans)]*valency)

       assert tdnode.make_logic(parents, adxs, out, args) == ans




###############################################################################
###                                                                         ###
###  Logic gates representing the logic of forward and reverse propagation. ###
###                                                                         ###
###############################################################################


# --- Forward gate ---------------------------------------------------------- #

class TestForwardGate(TestForward):

   def test_nodify(self):
 
       nodule = fake.Nodule()

       gate = self.gate()
       ans  = self.node(nodule, gate)

       assert gate.nodify(nodule) == ans  


   @pytest.mark.parametrize("rndseed", [1]) 
   def test_grad(self, rndseed):

       grad = self.randn(rndseed)
       gate = self.gate(grad=grad)

       assert value_eq(gate.grad(), grad)
       




# --- Forward node ---------------------------------------------------------- #

class TestForwardNode(TestForward): 

   @pytest.mark.parametrize("rndseed", [1])     
   def test_tovalue(self, rndseed):

       ans  = self.randn(rndseed)
       node = self.node(fake.Nodule(tovalue=ans))

       assert node.tovalue() == ans 

 
   def test_attach(self): 

       ans    = fake.NodeTrain()               
       train1 = fake.NodeTrain(with_meta=ans)  
       train2 = fake.NodeTrain(with_node=train1) 

       node = self.node(fake.Nodule(attach={train1: ans}))

       assert node.attach(train2) == ans


   @pytest.mark.parametrize("valency", [1, 2, 3]) 
   def test_logic(self, valency):

       adxs, out, args = self.args(valency)
       others          = tuple([fake.ForwardNode()]*(valency-1))

       node = self.node()
       ans  = self.logic((node, *others), adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

    
   @pytest.mark.parametrize("rndseed", [1])  
   def test_grad(self, rndseed):

       grad = self.randn(rndseed)
       node = self.node(gate=fake.ForwardGate(grad=grad))

       assert value_eq(node.grad(), grad)













































































