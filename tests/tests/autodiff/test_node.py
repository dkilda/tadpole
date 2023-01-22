#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.mocks           as mock
import tests.fixtures        as fixt
import tadpole.autodiff.node as tdnode

from tests.tests.common import value_eq

"""
from tests.fixtures.autodiff import (
                            randn,
                            jvpfun_args,
                            forward_logic,
                            forward_gate,
                            forward_node,
                           )
"""


from tests.fixtures.autodiff.misc import (
                                    randn,
                                    jvpfun_args,
                                   )
from tests.fixtures.autodiff.node import (
                                    forward_logic,
                                    forward_gate,
                                    forward_node,
                                   )




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
 
       nodule = mock.Nodule()

       gate = self.gate()
       ans  = self.node(nodule, gate)

       assert gate.nodify(nodule) == ans # Equality by id's / or compare repr # TODO: impl _signature(), use it for repr and equality  


   @pytest.mark.parametrize("rndseed", [1]) 
   def test_grad(self, rndseed):

       grad = self.randn(rndseed)
       gate = self.gate(grad=grad)

       assert value_eq(gate.grad(), grad)
       




class TestForwardNode: # FIXME rename mocks to fakes

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
       node = self.node(mock.Nodule(tovalue=ans))

       assert node.tovalue() == ans 

 
   def test_attach(self): 

       ans    = mock.NodeTrain()               
       train1 = mock.NodeTrain(with_meta=ans)  
       train2 = mock.NodeTrain(with_node=train1) 

       node = self.node(mock.Nodule(attach={train1: ans}))

       assert node.attach(train2) == ans


   @pytest.mark.parametrize("valency", [1, 2, 3]) 
   def test_logic(self, valency):

       adxs, out, args = self.args(valency)
       others          = tuple([mock.ForwardNode()]*(valency-1))

       node = self.node()
       ans  = self.logic((node, *others), valency, adxs, out, args)

       assert node.logic(others, adxs, out, args) == ans

    
   @pytest.mark.parametrize("rndseed", [1])  
   def test_grad(self, rndseed):

       grad = self.randn(rndseed)
       node = self.node(gate=mock.ForwardGate(grad=grad))

       assert value_eq(node.grad(), grad)













































































