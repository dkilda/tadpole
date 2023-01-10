#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from tests.common import assert_close

from tests.mocks.autodiff.node import MockNode, MockGate, MockGateInputsMock
from tests.mocks.autodiff.node import MockPoint
from tests.mocks.autodiff.node import ForwardNode, MockReverseNode 
from tests.mocks.autodiff.node import MockForwardGate, MockReverseGate

from tadpole.autodiff.node import Node, Gate, GateInputs
from tadpole.autodiff.node import Point
from tadpole.autodiff.node import UndirectedNode, ForwardNode, ReverseNode
from tadpole.autodiff.node import ForwardGate, ReverseGate
from tadpole.autodiff.node import ForwardGateInputs, ReverseGateInputs

from tadpole.autodiff.node import make_node



###############################################################################
###                                                                         ###
###  Fixtures                                                               ###
###                                                                         ###
###############################################################################


# --- Random value ---------------------------------------------------------- #

@pytest.fixture
def randn_val():

    def wrap(seed=1):

        np.random.seed(seed)    
        return np.random.randn()

    return wrap




# --- Nodes ----------------------------------------------------------------- #

@pytest.fixture
def forward_node():

    def wrap(source, gate, layer):

        return ForwardNode(UndirectedNode(source, gate, layer))

    return wrap




@pytest.fixture
def reverse_node():

    def wrap(source, gate, layer):

        return ReverseNode(UndirectedNode(source, gate, layer))

    return wrap




# --- Mock args ------------------------------------------------------------- #

def mock_args(node_type, randn_val):

    def wrap(n, adxs=None, vals=None):

        if adxs is None:
           adxs = list(range(n))

        if   vals is None:
             vals = (randn_val(i+1) for i in range(n))
        else:
             vals = iter(vals)

        return [node_type() if i in adxs else next(vals) for i in range(n)]

    return wrap




@pytest.fixture
def mock_forward_args(randn_val):

    return mock_args(MockForwardNode, randn_val)




@pytest.fixture
def mock_reverse_args(randn_val):

    return mock_args(MockReverseNode, randn_val)




# --- Default gates --------------------------------------------------------- #

@pytest.fixture
def default_forward_gate(randn_val):

    def wrap(valency=2, grad=None):

        if grad is None:
           grad = randn_val()

        parents = (MockForwardGate(), )*valency

        return ForwardGate(parents, grad)

    return wrap




@pytest.fixture
def default_reverse_gate(randn_val):

    def wrap(valency=2, vjp=None):

        if vjp is None:
           vjp = lambda g: (g * randn_val(i) for i in range(valency)) 

        parents = (MockReverseGate(), )*valency

        return ReverseGate(parents, vjp)

    return wrap




# --- Iterable of default gates --------------------------------------------- #

def default_gates(default_gate, randn_val):

    def wrap(n, valencies=None, grads=None):

        if valencies is None:
           valencies = [None]*n

        if grads is None:
           grads = [randn_val(i+1) for i in range(n)]

        return [default_gate(v, g) for v, g in zip(valencies, grads)]

    return wrap




@pytest.fixture
def default_forward_gates(default_forward_gate, randn_val):

    return default_gates(default_forward_gate, randn_val)




@pytest.fixture
def default_reverse_gates(default_reverse_gate, randn_val):

    return default_gates(default_reverse_gate, randn_val)




###############################################################################
###                                                                         ###
###  Gates of the autodiff circuit                                          ###
###                                                                         ###
###############################################################################


# --- Test forward gate ----------------------------------------------------- #

class TestForwardGate:


   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_randn_val(self, randn_val):

       self._randn_val = randn_val


   @pytest.fixture(autouse=True)
   def request_node(self, forward_node):

       self._node = forward_node


   @pytest.fixture(autouse=True)
   def request_default_gate(self, default_forward_gate):

       self._gate = default_forward_gate


   @pytest.fixture(autouse=True)
   def request_default_gates(self, default_forward_gates):

       self._gates = default_forward_gates


   @pytest.fixture(autouse=True)
   def request_mock_args(self, mock_forward_args):

       self._mock_args = mock_forward_args


   # --- Tests --- #

   @pytest.mark.parametrize("layer", [1])   
   def test_node(self, layer): 

       gate   = self._gate()
       source = MockForwardNode()
       layer  = 1

       out = gate.node(source, layer)
       ans = self._node(source, gate, layer) 

       assert out == ans


   @pytest.mark.parametrize("n, adxs, valencies",  
   [   
   [4, [0,2,3], [2,1,3,0]],   
   [1, [0],     [2]      ], 
   [2, [0],     [2]      ],                             
   ])   
   def test_next_input(self, n, adxs, valencies): 

       source = MockForwardNode()
       args   = self._mock_args(n, adxs)
       gates  = self._gates(n, valencies)

       out = gates[0].next_input(gates[1:], adxs, args, source)
       ans = ForwardGateInputs(gates, adxs, args, source) 

       assert out == ans


   @pytest.mark.parametrize("seed", [1, 2, 3])
   def test_grad(self, seed): 

       gate = self._gate(grad=self._randn_val(seed))

       assert_close(gate.grad(), grad) 



       












