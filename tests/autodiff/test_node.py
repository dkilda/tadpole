#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.node as tdnode
import tests.autodiff.fakes  as fake

from tests.autodiff.common import value_eq




# --- Forward gate ---------------------------------------------------------- #

class TestForwardGate:

   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_randn(self, randn):

       self.randn = randn


   @pytest.fixture(autouse=True)
   def request_gate(self, forward_gate):

       self.gate = forward_gate


   @pytest.fixture(autouse=True)
   def request_node(self, forward_node):

       self.node = forward_node


   # --- Tests --- #

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

class TestForwardNode: 

   # --- Fixtures --- #

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


   # --- Tests --- #

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
       ans  = self.logic((node, *others), valency, adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

    
   @pytest.mark.parametrize("rndseed", [1])  
   def test_grad(self, rndseed):

       grad = self.randn(rndseed)
       node = self.node(gate=fake.ForwardGate(grad=grad))

       assert value_eq(node.grad(), grad)













































































