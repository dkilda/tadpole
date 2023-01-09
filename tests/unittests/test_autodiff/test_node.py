#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from tadpole.tests.common import assert_close




class TestForwardGate:


   def _parents(nparents):

       return (MockForwardParents(),)*nparents


   def _source():

       return MockForwardNode()


   def _gate(self, nparents, grad):

       return ForwardGate(self._parents(nparents), grad)


   def test_node(self, layer, nparents, grad): 
    
       gate = self._gate(nparents, grad) 

       out = gate.node(self._source, layer)
       ans = ForwardNode(UndirectedNode(self._source, gate, layer))

       assert out == ans


   def test_next_input(self, adxs, args, nparents_list, grads): 

       elems = iter(zip(nparents_list, grads))

       this   = self._gate(*next(elems))
       others = [self._gate(*elem) for elem in elems]

       out = this.next_input(others, adxs, args, self._source)
       ans = ForwardGateInputs(gates, adxs, args, self._source) 

       assert out == ans


   def test_grad(self, nparents, grad):

       gate = self._gate(nparents, grad) 

       assert_close(gate.grad(), grad) # FIXME need to impl equality for objects and assert_allclose for floats















































