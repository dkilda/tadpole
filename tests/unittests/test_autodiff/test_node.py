#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from tests.common import assert_close

import tests.mocks.autodiff.node  as mknode
import tests.mocks.autodiff.graph as mkgraph
import tadpole.autodiff.node      as tdnode




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

        return tdnode.ForwardNode(tdnode.UndirectedNode(source, gate, layer))

    return wrap




@pytest.fixture
def reverse_node():

    def wrap(source, gate, layer):

        return tdnode.ReverseNode(tdnode.UndirectedNode(source, gate, layer))

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

    return mock_args(mknode.MockForwardNode, randn_val)




@pytest.fixture
def mock_reverse_args(randn_val):

    return mock_args(mknode.MockReverseNode, randn_val)




# --- Default gates --------------------------------------------------------- #

@pytest.fixture
def default_forward_gate(randn_val):

    def wrap(valency=2, grad=None):

        if grad is None:
           grad = randn_val()

        parents = (mknode.MockForwardGate(), )*valency

        return tdnode.ForwardGate(parents, grad)

    return wrap




@pytest.fixture
def default_reverse_gate(randn_val):

    def wrap(valency=2, vjp=None):

        if vjp is None:
           vjp = lambda g: (g * randn_val(i) for i in range(valency)) 

        parents = (mknode.MockReverseGate(), )*valency

        return tdnode.ReverseGate(parents, vjp)

    return wrap




# --- Iterable of default gates --------------------------------------------- #

def default_gates(default_gate, randn_val):

    def wrap(n, valencies=None, grads=None):

        if valencies is None:
           valencies = [None]*n

        if grads is None:
           grads = [randn_val(i+1) for i in range(n)]

        return tuple(default_gate(v, g) for v, g in zip(valencies, grads))

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


# --- Forward gate ---------------------------------------------------------- #

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
       source = mknode.MockForwardNode()
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

       gates  = self._gates(n, valencies)
       args   = self._mock_args(n, adxs)
       source = mkgraph.MockPack() 

       out = gates[0].next_input(gates[1:], adxs, args, source)
       ans = tdnode.ForwardGateInputs(gates, adxs, args, source) 

       assert out == ans


   @pytest.mark.parametrize("seed", [1, 2, 3])
   def test_grad(self, seed): 

       grad = self._randn_val(seed)
       gate = self._gate(grad=grad)

       assert_close(gate.grad(), grad) 




# --- Reverse gate inputs --------------------------------------------------- #

class TestReverseGateInputs:


   def test_transform(self):

       n = 1

       parents = self._gates(n)
       adxs    = tuple(range(n))
       args    = self._mock_args(n)
       source  = mkgraph.MockPack({fun: out})

       x   = self._randn_val()
 
       fun = tad.sin
       vjp = tdadj.VjpFactory(fun).vjp()

       # lambda g: tad.mul(g, tad.cos(x)) # FIXME even so it won't be the same vjp, cuz functions are retrieved by identity!

       # FIXME exclude vjp from comparisons? or use just vjp value?
     


       inputs = tdnode.ReverseGateInputs(parents, adxs, args, source) 

       out = inputs.transform(fun)
       ans = tdnode.ReverseGate(parents, vjp)

       assert out == ans






   def test_transform(self, fun, n, adx, valencies):

       gates   = self._gates(n, valencies)
       args    = self._mock_args(n, [adx])
       source  = mkgraph.MockPack() 

       fun = tad.sin
       vjp = lambda g: tad.mul(g, tad.cos(arg)) 


       inputs = tdnode.ReverseGateInputs(gates, adxs, args, source) 

       out = inputs.transform(fun)
       ans = tdnode.ReverseGate(parents, vjp)

       assert out == ans

       












































