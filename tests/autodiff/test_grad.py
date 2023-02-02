#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tadpole.autodiff.adjoints as tda
import tadpole.autodiff.util     as tdutil
import tadpole.autodiff.node     as tdnode
import tadpole.autodiff.graph    as tdgraph
import tadpole.autodiff.grad     as tdgrad
import tests.autodiff.fakes      as fake

import tests.common.ntuple as tpl





# TODO probs we should just use factories for fakes, and ctors/fixtures for real objects?


class ReverseGateFactory: # TODO could also make SeededReverseGateFactory decorator...

   def __init__(self, parents=tuple(), seed=None):

       if isinstance(parents, int):
          parents = tpl.repeat(fake.ReverseNode, parents)

       self._parents = parents
       self._seed    = seed


   @property
   @tdutil.cacheable
   def gradfun(self):

       if self._seed is None:
          return lambda seed: self.grads

       return lambda seed: {self._seed: self.grads}[seed]


   @property
   @tdutil.cacheable
   def parents(self):

       return self._parents


   @property
   @tdutil.cacheable
   def grads(self):

       return tpl.repeat(fake.CumFunReturn, self.valency)


   @property
   @tdutil.cacheable
   def valency(self):

       return len(self._parents)


   @property
   @tdutil.cacheable
   def gate(self):

       return fake.ReverseGate(
                               parents=self.parents, 
                               grads=self.gradfun
                              )  







class ReverseNodeFactory:

   def __init__(self, gatefactory):

       self._gatefactory = gatefactory


   @property
   @tdutil.cacheable
   def parents(self):

       return self._gatefactory.parents


   @property
   @tdutil.cacheable
   def grads(self):

       return self._gatefactory.grads


   @property
   @tdutil.cacheable
   def valency(self):

       return self._gatefactory.valency


   @property
   @tdutil.cacheable
   def gate(self):

       return self._gatefactory.gate


   @property
   @tdutil.cacheable
   def node(self):  

       return fake.ReverseNode(gate=self.gate)




@pytest.fixture
def reverse_gate_factory():

    def wrap(*args, **kwargs):
        return ReverseGateFactory(*args, **kwargs)

    return wrap




@pytest.fixture
def reverse_node_factory(reverse_gate_factory):

    def wrap(*args, **kwargs):

        gatefactory = reverse_gate_factory(*args, **kwargs)

        return ReverseNodeFactory(gatefactory)

    return wrap













"""

class ReverseGateFactory:

   def __init__(self, parents=None, seed=None):

       if parents is None: parents = 2
       if seed    is None: seed    = fake.FunReturn()

       self._parents = parents
       self._seed    = seed


   def with_seed(self, seed):

       return self.__class__(self._parents, seed)


   @property
   @tdutil.cacheable
   def valency(self):

       return len(self.parents)


   @property
   @tdutil.cacheable
   def parents(self):

       if isinstance(self._parents, int):
          return tpl.repeat(fake.FunReturn, self._parents) # FIXME: parents can be int, list of FunReturn, list of Nodes

       return self._parents


   @property
   @tdutil.cacheable
   def grads(self):

       return tpl.repeat(fake.FunReturn, self.valency)


   @property
   @tdutil.cacheable
   def seed(self):

       return self._seed


   @property
   @tdutil.cacheable
   def fun(self):

       return fake.Fun()


   @property
   @tdutil.cacheable
   def vjp(self):

       return fake.Fun({(self.seed,): self.grads})


   @property
   @tdutil.cacheable
   def gate(self):

       return tdnode.ReverseGate(self.parents, self.fun, self.vjp)  




class ReverseNodeFactory:

   def __init__(self, gatefactory, source=None, layer=None):

       if source is None: source = fake.FunReturn()
       if layer  is None: layer  = fake.FunReturn()

       self._gatefactory = gatefactory
       self._source      = source
       self._layer       = layer


   def with_seed(self, seed):

       return self.__class__(self._gatefactory.with_seed(seed), source, layer)


   @property
   @tdutil.cacheable   
   def source(self):

       return self._source  


   @property
   @tdutil.cacheable   
   def layer(self):

       return self._layer  


   @property
   @tdutil.cacheable   
   def nodule(self):

       return tdnode.Nodule(self.source, self.layer)


   @property
   @tdutil.cacheable
   def valency(self):

       return self._gatefactory.valency


   @property
   @tdutil.cacheable
   def parents(self):

       return self._gatefactory.parents


   @property
   @tdutil.cacheable
   def grads(self):

       return self._gatefactory.grads 


   @property
   @tdutil.cacheable   
   def gate(self):

       return self._gatefactory.gate 

 
   @property
   @tdutil.cacheable
   def node(self):

       return tdnode.ReverseNode(self.nodule, self.gate)  




@pytest.fixture
def reverse_gate_factory():

    def wrap(*args, **kwargs):
        return ReverseGateFactory(*args, **kwargs)

    return wrap




@pytest.fixture
def reverse_node_factory(reverse_gate_factory):

    def wrap(parents=None, source=None, layer=None):

        gatefactory = reverse_gate_factory(parents)

        return ReverseNodeFactory(gatefactory, source, layer)

    return wrap
"""




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


   @pytest.mark.parametrize("max_valency", [3]) 
   def test_iterate(self, max_valency): 

       parents = tpl.repeat(self._node, max_valency)

       nodeA = self._node((parents[0], parents[1])) 
       nodeB = self._node((parents[0], parents[2]))    
       nodeC = self._node((nodeA,))  
       nodeD = self._node((nodeB, parents[1]))
       nodeE = self._node((nodeC, parents[1], nodeD))  

       ans = (nodeE, nodeD, nodeB, parents[2], nodeC, 
              nodeA, parents[1], parents[0])

       assert tuple(tdgrad.toposort(nodeE)) == ans 




# --- Gradient accumulation ------------------------------------------------- #

class TestGradAccum:

   @pytest.fixture(autouse=True)
   def request_gradaccum(self, gradaccum):

       self.gradaccum = gradaccum


   def test_push(self): # FIXME cannot test .push() in isolation from .pop()

       node = fake.ReverseNode()
       grad = fake.FunReturn()

       gradaccum = self.gradaccum()
       assert gradaccum.push(node, grad) == gradaccum

 
   def test_pop(self): # FIXME cannot test .push() in isolation from .pop()

       node = fake.ReverseNode()
       grad = fake.FunReturn()

       gradaccum = self.gradaccum()
       gradaccum.push(node, grad) 
      
       assert gradaccum.pop(node) == grad


   def test_result(self): # FIXME cannot test .result() in isolation from .push() and .pop()

       node = fake.ReverseNode()
       grad = fake.FunReturn()

       gradaccum = self.gradaccum()
       gradaccum.push(node, grad) 

       out = gradaccum.pop(node)
      
       assert gradaccum.result() == out


   def test_result_simple(self): # FIXME cannot test .result() in isolation from .push() and .pop()

       node = fake.ReverseNode()
       grad = fake.FunReturn()

       gradaccum = self.gradaccum()      
       assert gradaccum.result() == None # FIXME should not return None! return a zero grad instead!


   def test_accumulate(self):

       nodeA, gradA = fake.ReverseNode(), fake.CumFunReturn()
       nodeB, gradB = fake.ReverseNode(), fake.CumFunReturn()
       nodeC, gradC = fake.ReverseNode(), fake.CumFunReturn()

       gradaccum = self.gradaccum()   
       gradaccum.accumulate(nodeA, gradA)
       gradaccum.accumulate(nodeB, gradB)
       gradaccum.accumulate(nodeA, gradB)
       gradaccum.accumulate(nodeC, gradC)
       gradaccum.accumulate(nodeB, gradA)
       gradaccum.accumulate(nodeB, gradC)       

       assert gradaccum.pop(nodeA) == (gradA + gradB)
       assert gradaccum.pop(nodeB) == (gradB + gradA + gradC)
       assert gradaccum.pop(nodeC) == gradC


   @pytest.mark.parametrize("valency", [0,1,2,3])
   def test_accumulate_simple(self, valency):

       nodes = tpl.repeat(fake.ReverseNode, valency)
       grads = tpl.repeat(fake.FunReturn, valency)

       gradaccum = self.gradaccum()
       for node, grad in zip(nodes, grads):
           gradaccum.accumulate(node, grad)

       assert tpl.amap(gradaccum.pop, nodes) == grads 
       
       
       

# --- Backpropagation ------------------------------------------------------- #

class TestBackprop:

   @pytest.fixture(autouse=True)
   def request_backprop(self, backprop):

       self.backprop = backprop


   @pytest.fixture(autouse=True)
   def request_nodefactory(self, reverse_node_factory):

       self.nodefactory = reverse_node_factory


   def test_call(self):

       x0 = self.nodefactory()
       x1 = self.nodefactory()
       x2 = self.nodefactory()

       xA = self.nodefactory((x0.node, x1.node))
       xB = self.nodefactory((x0.node, x2.node))
       xC = self.nodefactory((xA.node,        ))
       xD = self.nodefactory((xB.node, x1.node))
       xE = self.nodefactory((xC.node, x1.node, xD.node))

       backprop = self.backprop(xE.node)

       assert backprop(fake.CumFunReturn()) == (xA.grads[0] + xB.grads[0])
 



   """

   def _node(self, parents=tuple(), grads=tuple(), which=tuple()):

       grads = tuple(grads[i] for i in which)
       gate  = fake.ReverseGate(parents=parents, grads=grads)
       return fake.ReverseNode(gate=gate)
   
    
   def _grads(self, *parents):

       return {parent: fake.CumFunReturn() for parent in parents}


   def test_call(self):

       gradsA = self._grads(0,1)
       gradsB = self._grads(0,2)
       gradsC = self._grads("A")
       gradsD = self._grads("B",1)
       gradsE = self._grads("C","D",1)

       leaf0 = self._node()
       leaf1 = self._node()
       leaf2 = self._node()
       nodeA = self._node((leaf0, leaf1),        gradsA, (0,   1))
       nodeB = self._node((leaf0, leaf2),        gradsB, (0,   2))
       nodeC = self._node((nodeA,      ),        gradsC, ("A",  ))
       nodeD = self._node((nodeB, leaf1),        gradsD, ("B", 1))
       nodeE = self._node((nodeC, leaf1, nodeD), gradsE, ("C", 1, "D"))

       backprop = self.backprop(nodeE)

       assert backprop(fake.CumFunReturn()) == (gradsA[0] + gradsB[0])
   """




























