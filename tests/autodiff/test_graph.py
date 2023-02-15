#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

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


   # --- Test enter and build --- #

   @pytest.mark.parametrize("which", ["REVERSE", "FORWARD"])
   def test_enter(self, which):

       dat = data.graph_dat(which, 0)

       with tdgraph.Graph(dat.root) as graph:
          assert graph.build(dat.fun, dat.x) == dat.end


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
       w = data.concat_args_dat(x)

       assert x.args.concat() == w.concat

       
   @pytest.mark.parametrize("n, adxs, layers", [
      [1, (0,),  (0,)],
      [2, (0,),  (1,)],
      [2, (1,),  (0,)],
      [2, (0,1), (0,0)],
      [3, (0,2), (0,1)],
   ])  
   def test_pack(self, n, adxs, layers):

       x = data.args_dat(n, adxs, layers)
       w = data.pack_args_dat(x)

       assert x.args.pack() == w.pack


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
           print("CONTAINS-3(loop): ", node, x.args._args)
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

       x = data.args_dat(n, adxs, layers)

       nodes, sources, layers = x.nodes[:-1], x.sources[:-1], x.layers[:-1]
       node,  source,  layer  = x.nodes[-1],  x.sources[-1],  x.layers[-1]

       w1 = data.concat_dat(nodes, sources, layers)
       w2 = data.concat_dat(x.nodes, x.sources, x.layers)

       assert w1.concat.attach(node, source, layer) == w2.concat     


   @pytest.mark.parametrize("n, adxs, layers, result", [
      [1, (0,),    (0,),     0],
      [2, (0,),    (1,),     1],
      [2, (1,),    (0,),     0],
      [2, (0,1),   (0,0),    0],
      [2, (0,1),   (0,2),    2],
      [3, (0,2),   (0,1),    1],
      [3, (0,2),   (2,1),    2],
      [2, tuple(), tuple(), -1],
   ])  
   def test_layer(self, n, adxs, layers, result):

       w = data.concat_args_dat(data.args_dat(n, adxs, layers))

       assert w.concat.layer() == result
  

   @pytest.mark.parametrize("n, adxs, layers, result", [
      [1, (0,),    (0,),    (0,)   ],
      [2, (0,),    (1,),    (0,)   ],
      [2, (1,),    (0,),    (1,)   ],
      [2, (0,1),   (0,0),   (0,1)  ],
      [2, (0,1),   (0,2),   (1,)   ],
      [3, (0,2),   (0,1),   (2,)   ],
      [3, (0,2),   (2,1),   (0,)   ],
      [3, (0,2),   (1,1),   (0,2)  ],
      [2, tuple(), tuple(), tuple()],
   ])  
   def test_adxs(self, n, adxs, layers, result):

       w = data.concat_args_dat(data.args_dat(n, adxs, layers))

       assert w.concat.adxs() == result


   @pytest.mark.parametrize("n, adxs, layers, which", [
      [1, (0,),    (0,),    0],
      [2, (0,),    (1,),    1],
      [2, (1,),    (0,),    2],
      [2, (0,1),   (0,0),   3],
      [2, (0,1),   (0,2),   4],
      [3, (0,2),   (0,1),   5],
      [3, (0,2),   (2,1),   6],
      [3, (0,2),   (1,1),   7],
      [2, tuple(), tuple(), 8],
   ])  
   def tests_parents(self, n, adxs, layers, which):

       x = data.args_dat(n, adxs, layers)
       w = data.concat_args_dat(x)

       result = (
                 lambda: (x.nodes[0], ),
                 lambda: (x.nodes[0], ),
                 lambda: (x.nodes[1], ),
                 lambda: (x.nodes[0], x.nodes[1]),
                 lambda: (x.nodes[1], ),
                 lambda: (x.nodes[2], ),
                 lambda: (x.nodes[0], ),
                 lambda: (x.nodes[0], x.nodes[2]),
                 lambda: tuple(),
                )[which]()

       assert w.concat.parents() == tdnode.Parents(result)


   @pytest.mark.parametrize("n, adxs, layers, which", [
      [1, (0,),    (0,),    0],
      [2, (0,),    (1,),    1],
      [2, (1,),    (0,),    2],
      [2, (0,1),   (0,0),   3],
      [2, (0,1),   (0,2),   4],
      [3, (0,2),   (0,1),   5],
      [3, (0,2),   (2,1),   6],
      [3, (0,2),   (1,1),   7],
      [2, tuple(), tuple(), 8],
   ])  
   def tests_deshell(self, n, adxs, layers, which):

       x = data.args_dat(n, adxs, layers)
       w = data.concat_args_dat(x)       

       result = (
                 lambda: (x.sources[0],                             ),
                 lambda: (x.sources[0], x.nodes[1],                 ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.sources[0], x.sources[1],               ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.nodes[0],   x.nodes[1],   x.sources[2], ),
                 lambda: (x.sources[0], x.nodes[1],   x.nodes[2],   ),
                 lambda: (x.sources[0], x.nodes[1],   x.sources[2], ),
                 lambda: (x.nodes[0],   x.nodes[1],                 ),
                )[which]()

       assert w.concat.deshell() == tdgraph.Args(result)




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack (of concatenated nodes) --------------------------------- #

class TestPack:

   @pytest.mark.parametrize("n, adxs, layers, result", [
      [1, (0,),    (0,),     False],
      [2, (0,),    (1,),     False],
      [2, (1,),    (0,),     False],
      [2, (0,1),   (0,0),    False],
      [2, (0,1),   (0,2),    False],
      [3, (0,2),   (0,1),    False],
      [3, (0,2),   (2,1),    False],
      [2, tuple(), tuple(),  True],
   ])  
   def test_innermost(self, n, adxs, layers, result):

       x = data.args_dat(n, adxs, layers)
       w = data.pack_args_dat(x)  

       assert w.pack.innermost() == result


   @pytest.mark.parametrize("n, adxs, layers, which", [
      [1, (0,),    (0,),    0],
      [2, (0,),    (1,),    1],
      [2, (1,),    (0,),    2],
      [2, (0,1),   (0,0),   3],
      [2, (0,1),   (0,2),   4],
      [3, (0,2),   (0,1),   5],
      [3, (0,2),   (2,1),   6],
      [3, (0,2),   (1,1),   7],
      [2, tuple(), tuple(), 8],
   ])  
   def tests_deshell(self, n, adxs, layers, which):

       x = data.args_dat(n, adxs, layers)
       w = data.pack_args_dat(x)  

       result = (
                 lambda: (x.sources[0],                             ),
                 lambda: (x.sources[0], x.nodes[1],                 ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.sources[0], x.sources[1],               ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.nodes[0],   x.nodes[1],   x.sources[2], ),
                 lambda: (x.sources[0], x.nodes[1],   x.nodes[2],   ),
                 lambda: (x.sources[0], x.nodes[1],   x.sources[2], ),
                 lambda: (x.nodes[0],   x.nodes[1],                 ),
                )[which]()

       assert w.pack.deshell() == tdgraph.Args(result)


   @pytest.mark.parametrize("n, adxs, layers, which", [
      [1, (0,),    (0,),    0],
      [2, (0,),    (1,),    1],
      [2, (1,),    (0,),    2],
      [2, (0,1),   (0,0),   3],
      [2, (0,1),   (0,2),   4],
      [3, (0,2),   (0,1),   5],
      [3, (0,2),   (2,1),   6],
      [3, (0,2),   (1,1),   7],
      [2, tuple(), tuple(), 8],
   ])  
   def test_deshelled(self, n, adxs, layers, which):

       x = data.args_dat(n, adxs, layers)
       w = data.pack_args_dat(x)  

       result = (
                 lambda: (x.sources[0],                             ),
                 lambda: (x.sources[0], x.nodes[1],                 ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.sources[0], x.sources[1],               ),
                 lambda: (x.nodes[0],   x.sources[1],               ),
                 lambda: (x.nodes[0],   x.nodes[1],   x.sources[2], ),
                 lambda: (x.sources[0], x.nodes[1],   x.nodes[2],   ),
                 lambda: (x.sources[0], x.nodes[1],   x.sources[2], ),
                 lambda: (x.nodes[0],   x.nodes[1],                 ),
                )[which]()

       pack = tdgraph.Args(result).pack()

       assert w.pack.deshelled() == pack




   @pytest.mark.parametrize("layer, result", [
      [-1, True], 
      [ 0, False], 
      [ 1, False], 
      [ 2, False],
   ])
   def test_innermost_001(self, layer, result):

       concat = fake.Cohesive(layer=fake.Fun(layer))
       pack   = tdgraph.Pack(concat)
       assert pack.innermost() == result


   def test_deshell_001(self):

       args   = fake.ArgsLike()
       concat = fake.Cohesive(deshell=fake.Fun(args))

       pack = tdgraph.Pack(concat)
       assert pack.deshell() == args


   def test_deshelled_001(self):

       out    = fake.Packable()
       args   = fake.ArgsLike(pack=fake.Fun(out))
       concat = fake.Cohesive(deshell=fake.Fun(args))

       pack = tdgraph.Pack(concat)
       assert pack.deshelled() == out


   @pytest.mark.parametrize("valency", [1,2,3])
   @pytest.mark.parametrize("layer",   [0,1])
   def test_fold_001(self, valency, layer):
 
       fun   = fake.Fun(None)
       adxs  = common.arepeat(fake.Value, valency)
       out   = fake.NodeLike()
       args  = fake.ArgsLike()

       op      = tdnode.AdjointOp(fun, adxs, out, args)
       node    = fake.NodeLike()
       parents = fake.Parental(next=fake.Fun(node, out, layer, op))  

       concat = fake.Cohesive(
                              layer=fake.Fun(layer),
                              adxs=fake.Fun(adxs),
                              parents=fake.Fun(parents),
                              deshell=fake.Fun(args)
                             )
       pack = tdgraph.Pack(concat)

       assert pack.fold(fun, out) == node 


   def test_fold_002(self):
 
       fun   = fake.Fun(None)
       out   = fake.Value()
       point = tdnode.Point(out)

       concat = fake.Cohesive(layer=fake.Fun(tdgraph.minlayer()))
       pack   = tdgraph.Pack(concat)

       assert pack.fold(fun, out) == point       






















