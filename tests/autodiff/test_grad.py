#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.node     as tdnode
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.grad     as tdgrad
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Forward differential operator ----------------------------------------- #

class TestForwardDiffOp:

   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_diff_op(self, forward_diff_op):

       self.diff_op = forward_diff_op


   @pytest.fixture(autouse=True)
   def request_gate(self, forward_gate):

       self.gate = forward_gate


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.fixture(autouse=True)
   def request_node(self, forward_node):

       self.node = forward_node


   @pytest.fixture(autouse=True)
   def request_point(self, point):

       self.point = point


   # --- Setup --- #

   def _setup_helper(self, x, layer=0, seed=1, val=None, grad=None): # FIXME absorb into fixture?

       if val  is None: val  = fake.FunReturn()
       if grad is None: grad = fake.FunReturn()

       def fun(x):
           if isinstance(x, fake.FunReturn): 
              return self.point(x)
           return x

       xnode = self.node(
                         self.nodule(fun(x), layer), 
                         tdnode.ForwardRootGate(seed)  # FIXME ideally we could just pass a fake instead of 
                        )                              # having to use a real object...

       ansgate = fake.ForwardGate(grad=grad)
       ansnode = fake.ForwardNode(tovalue=val, gate=ansgate)

       return xnode, ansnode 


   def _setup(self, x, *args_, **kwargs_):

       xnode, ansnode = self._setup_helper(x, *args_, **kwargs_)

       return fake.Fun(fake.Map({(xnode,): ansnode}))


   def _setup_nary(self, args, adx, *args_, **kwargs_):

       x = args[adx]

       xnode, ansnode = self._setup_helper(x, *args_, **kwargs_)

       args1 = tuple(xnode if i == adx else arg 
                           for i, arg in enumerate(args))

       return fake.Fun(fake.Map({args1: ansnode}))


   # --- Tests --- #

   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])  
   def test_evaluate(self, xtype):

       x   = xtype()
       ans = xtype()
       fun = self._setup(x, val=ans)

       op = self.diff_op(fun, x)

       assert op.evaluate() == ans


   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])   
   @pytest.mark.parametrize("seed",  [0, 0.5, 1.0, -2.3])   
   def test_grad(self, xtype, seed):

       x   = xtype()
       ans = fake.FunReturn()
       fun = self._setup(x, seed=seed, grad=ans)

       op = self.diff_op(fun, x)

       assert op.grad(seed) == ans 


   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])   
   @pytest.mark.parametrize("seed",  [0, 0.5, 1.0, -2.3])  
   def test_evaluate_and_grad(self, xtype, seed):    

       x    = xtype()
       val  = xtype()
       grad = fake.FunReturn()
       fun  = self._setup(x, seed=seed, val=val, grad=grad)

       op = self.diff_op(fun, x)

       assert op.evaluate_and_grad(seed) == (val, grad) 


   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])
   def test_deriv(self, xtype):

       x   = xtype()
       ans = fake.FunReturn()
       fun = self._setup(x, grad=ans)

       assert tdgrad.deriv(fun)(x) == ans


   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])  
   @pytest.mark.parametrize("valency, adx", [
      [1,0], 
      [2,0], 
      [2,1],
      [2,1], 
      [3,0], 
      [3,1], 
      [3,2],
   ])  
   def test_nary_deriv(self, xtype, valency, adx):

       args = tpl.repeat(xtype, valency)
       ans  = fake.FunReturn()
       fun  = self._setup_nary(args, adx, grad=ans)

       assert tdgrad.deriv(fun, adx)(*args) == ans




# --- Reverse differential operator ----------------------------------------- #

class TestReverseDiffOp:

   # --- Fixtures --- #

   @pytest.fixture(autouse=True)
   def request_diff_op(self, reverse_diff_op):

       self.diff_op = reverse_diff_op


   @pytest.fixture(autouse=True)
   def request_gate(self, reverse_gate):

       self.gate = reverse_gate


   @pytest.fixture(autouse=True)
   def request_nodule(self, nodule):

       self.nodule = nodule


   @pytest.fixture(autouse=True)
   def request_node(self, reverse_node):

       self.node = reverse_node


   @pytest.fixture(autouse=True)
   def request_point(self, point):

       self.point = point


   # --- Setup --- #

   def _setup_helper(self, x, layer=0, val=None, parents=None, grads=None): # FIXME absorb into fixture?

       if val     is None: val     = fake.FunReturn()
       if grads   is None: grads   = tpl.repeat(fake.FunReturn,   2)
       if parents is None: parents = tpl.repeat(fake.ReverseNode, 2)

       def fun(x):
           if isinstance(x, fake.FunReturn): 
              return self.point(x)
           return x

       xnode = self.node(
                         self.nodule(fun(x), layer), 
                         tdnode.ReverseRootGate()  # FIXME ideally we could just pass a fake instead of 
                        )                          # having to use a real object...

       ansgate = fake.ReverseGate(parents=parents, grads=grads)
       ansnode = fake.ReverseNode(tovalue=val, gate=ansgate)

       return xnode, ansnode # FIXME: a major issue: we have to mock out an entire Backprop to test this!
                             # This makes .grad() very difficult to test. Instead, we must find a way to 
                             # supply a fake Backprop!

   def _setup(self, x, *args_, **kwargs_):

       xnode, ansnode = self._setup_helper(x, *args_, **kwargs_)

       return fake.Fun(fake.Map({(xnode,): ansnode}))


   # --- Tests --- #

   @pytest.mark.parametrize("xtype", [fake.FunReturn, fake.ForwardNode])  
   def test_evaluate(self, xtype):

       x   = xtype()
       ans = xtype()
       fun = self._setup(x, val=ans)

       op = self.diff_op(fun, x)

       assert op.evaluate() == ans




###############################################################################
###                                                                         ###
###  Backpropagation through the computation graph.                         ###
###                                                                         ###
###############################################################################


# --- Child-node counter ---------------------------------------------------- #

class TestChildCount:

   @pytest.fixture(autouse=True)
   def request_childcount(self, childcount):

       self.childcount = childcount


   def _node(self, parents=tuple()): 

       return fake.ReverseNode(gate=fake.ReverseGate(parents=parents))


   @pytest.mark.parametrize("nvisits", [1,2,3]) # FIXME cannot test .visit() in isolation from .iterate()
   def test_visit(self, nvisits):

       childcount = self.childcount()
       node       = fake.ReverseNode()

       for _ in range(nvisits):
           childcount.visit(node)

       assert dict(childcount.iterate())[node] == nvisits


   @pytest.mark.parametrize("valency", [1,2,3]) # FIXME cannot test .add() in isolation from .iterate()
   def test_add(self, valency):

       childcount = self.childcount()
       nodes      = tpl.repeat(fake.ReverseNode, valency)

       before = dict(childcount.iterate())
       childcount.add(nodes)                # FIXME there is no way to test the actual contents of ._pool!
       after  = dict(childcount.iterate())  # cuz _pool is a completely hidden internal state...
                                            # One idea: extract pool logic to a separate class.
       assert before == after


   @pytest.mark.parametrize("valency", [1,2,3]) 
   def test_compute(self, valency): # FIXME cannot test .compute() in isolation from .iterate() 

       parents  = tpl.repeat(self._node, valency)
       top_node = self._node(parents) 

       ans           = dict(zip(parents, [1]*valency))
       ans[top_node] = 1

       childcount = self.childcount(top_node)
       childcount.compute()     

       assert dict(childcount.iterate()) == ans


   @pytest.mark.parametrize("max_valency", [3]) 
   def test_compute_advanced(self, max_valency): # FIXME cannot test .compute() in isolation from .iterate()

       parents = tpl.repeat(self._node, max_valency)

       nodeA = self._node((parents[0], parents[1])) 
       nodeB = self._node((parents[0], parents[2]))    
       nodeC = self._node((nodeA,))  
       nodeD = self._node((nodeB, parents[1]))
       nodeE = self._node((nodeC, parents[1], nodeD))   

       ans = {
              parents[0]: 2,
              parents[1]: 3,
              parents[2]: 1,
              nodeA: 1,
              nodeB: 1,
              nodeC: 1,
              nodeD: 1,
              nodeE: 1,
             }

       childcount = self.childcount(nodeE)
       childcount.compute()     

       assert dict(childcount.iterate()) == ans




# --- Topological sort ------------------------------------------------------ #

class TestTopoSort:

   @pytest.fixture(autouse=True)
   def request_toposort(self, toposort):

       self.toposort = toposort


   def _node(self, parents=tuple()): 

       return fake.ReverseNode(gate=fake.ReverseGate(parents=parents))


   def test_add(self): # FIXME cannot test .add() in isolation from .iterate()

       node = self._node()

       toposort = self.toposort({node: 1}, node)
       toposort.add(node)

       assert tuple(toposort.iterate()) == (node,)


   @pytest.mark.parametrize("max_valency", [3])  # FIXME but: this time we can test .iterate() in isolation from .add()!
   def test_iterate(self, max_valency): 

       parents = tpl.repeat(self._node, max_valency)

       nodeA = self._node((parents[0], parents[1])) 
       nodeB = self._node((parents[0], parents[2]))    
       nodeC = self._node((nodeA,))  
       nodeD = self._node((nodeB, parents[1]))
       nodeE = self._node((nodeC, parents[1], nodeD))  

       count = {
                parents[0]: 2,
                parents[1]: 3,
                parents[2]: 1,
                nodeA: 1,
                nodeB: 1,
                nodeC: 1,
                nodeD: 1,
                nodeE: 1,
               }

       toposort = self.toposort(count, nodeE)
       ans      = (nodeE, nodeD, nodeB, parents[2], nodeC, 
                   nodeA, parents[1], parents[0])

       assert tuple(toposort.iterate()) == ans 

















