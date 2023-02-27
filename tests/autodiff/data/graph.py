#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.autodiff.node  as anode
import tadpole.autodiff.graph as agraph
import tadpole.autodiff.grad  as agrad
import tadpole.util           as util




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

GraphData = collections.namedtuple("GraphData", [
               "graphop", "graph", "layer", "root", "fun",
               "x", "out", "start", "end",
            ])




def graph_dat(which="REVERSE", layer=None, 
              x=None, out=None, start=None, end=None):

    root = {
            "REVERSE": anode.ReverseGate,
            "FORWARD": anode.ForwardGate,
           }[which]()

    if layer is None:
       layer = 0

    if x is None:
       x = fake.Value()

    if out is None:
       out = fake.Value()

    if start is None:
       start = anode.Node(x, layer, root) 

    if end is None:
       src = fake.NodeLike()
       end = anode.Node(src, layer, fake.GateLike())

    fun     = fake.Fun(end, start)
    graph   = agraph.Graph(root)
    graphop = agrad.GraphOp(root, fun, x)

    return GraphData(graphop, graph, layer, root, fun, 
                     x, out, start, end)




###############################################################################
###                                                                         ###
###  Autodiff function wrappers                                             ###
###  for differentiable and non-differentiable functions                    ###
###                                                                         ###
###############################################################################


# --- Differentiable/NonDifferentiable function wrappers -------------------- #

FunWrapData = collections.namedtuple("FunWrapData", [
                 "funwrap", "fun", 
                 "make_envelope", "envelope", 
                 "args", "out"
              ])




def differentiable_funwrap_dat(args):

    out = fake.NodeLike()

    applywrap     = fake.Fun(out)
    envelope      = fake.EnvelopeLike(applywrap=applywrap)
    make_envelope = fake.Fun(envelope, *args)

    fun     = fake.Fun(None)
    funwrap = agraph.Differentiable(fun, make_envelope)

    applywrap.update_args(funwrap, fun)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




def nondifferentiable_funwrap_dat(args):

    out = fake.NodeLike()

    envelope      = fake.EnvelopeLike(**{"apply": fake.Fun(out)})
    make_envelope = fake.Fun(envelope, *args)

    fun     = fake.Fun(None)
    funwrap = agraph.NonDifferentiable(fun, make_envelope)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation.                            ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

ArgsData = collections.namedtuple("ArgsData", [
              "args", "concat", "pack",
              "adxs", 
              "nodes", "sources", "layers", "values", "rawargs",
           ])




def args_dat(n=1, adxs=(0,), layers=(0,)):

    info   = []
    layers = list(reversed(layers))

    for i in range(n):

        value = fake.Value()

        if   i in adxs: 
             source = fake.NodeLike()
             layer  = layers.pop()
             node   = anode.Node(source, layer, fake.GateLike())
             rawarg = node

        else:
             source = value
             layer  = agraph.minlayer()           
             node   = anode.Point(value)
             rawarg = value

        info.append([rawarg, node, source, layer, value])

    rawargs, nodes, sources, layers, values = zip(*info)
    args = agraph.Args(*rawargs)

    concat = agraph.Concatenation(
                                   util.Sequence(nodes), 
                                   util.Sequence(sources), 
                                   util.Sequence(layers)
                                  )
    pack = agraph.Pack(concat)

    return ArgsData(args, concat, pack,
                    adxs, nodes, sources, layers, values, rawargs)




def nodeargs_dat(n=1, layers=None):

    if layers is None:
       layers = arepeat(0, n)

    return args_dat(n, range(n), layers)



      
# --- Concatenation of nodes ------------------------------------------------ #

ConcatData = collections.namedtuple("ConcatData", [
                "concat", "concat_history", "attach_history", 
                "nodes", "sources", "layers",
             ])




def concat_dat(n, adxs, layers):

    x = args_dat(n, adxs, layers)

    attach_history = list(zip(x.nodes, x.sources, x.layers))
    concat_history = []

    for size in range(n):

        _nodes   = util.Sequence(x.nodes[:size])
        _sources = util.Sequence(x.sources[:size])
        _layers  = util.Sequence(x.layers[:size])

        _concat  = agraph.Concatenation(_nodes, _sources, _layers)

        concat_history.append(_concat)
         
    nodes   = util.Sequence(x.nodes)
    sources = util.Sequence(x.sources)
    layers  = util.Sequence(x.layers)

    concat  = agraph.Concatenation(nodes, sources, layers)

    return ConcatData(concat, concat_history, attach_history, 
                      nodes, sources, layers)




ConcatOutputData = collections.namedtuple("ConcatOutputData", [
                      "concat", "layer", "adxs", "parents", "deshell",
                   ])




def concat_output_dat(n, adxs, layers):

    x = args_dat(n, adxs, layers)

    def case_000():
        return (
           0, (0,  ), (x.nodes[0], ), (x.sources[0], )
        )

    def case_001():
        return (
           1, (0,  ), (x.nodes[0], ), (x.sources[0], x.nodes[1])
        )

    def case_002():
        return (
           0, (1,  ), (x.nodes[1], ), (x.nodes[0], x.sources[1])
        )

    def case_003():
        return (
           0, (0, 1), (x.nodes[0], x.nodes[1]), (x.sources[0], x.sources[1])
        )

    def case_004():
        return (
           2, (1,  ), (x.nodes[1], ), (x.nodes[0], x.sources[1])
        )

    def case_005():
        return (
           1, (2,  ), (x.nodes[2], ), (x.nodes[0], x.nodes[1], x.sources[2])
        )

    def case_006():
        return (
           2, (0,  ), (x.nodes[0], ), (x.sources[0], x.nodes[1], x.nodes[2])
        )

    def case_007():
        return (
           1, (0, 2), (x.nodes[0], x.nodes[2]), (x.sources[0], x.nodes[1], x.sources[2])
        )

    def case_008():
        return (
           -1, tuple(), tuple(), (x.sources[0], x.sources[1])
        )

    case = {
            (1,  (0,),    (0,)    ): case_000,
            (2,  (0,),    (1,)    ): case_001,
            (2,  (1,),    (0,)    ): case_002,
            (2,  (0,1),   (0,0)   ): case_003,
            (2,  (0,1),   (0,2)   ): case_004,
            (3,  (0,2),   (0,1)   ): case_005,
            (3,  (0,2),   (2,1)   ): case_006,
            (3,  (0,2),   (1,1)   ): case_007,
            (2,  tuple(), tuple() ): case_008,
           }[n, adxs, layers]()


    return ConcatOutputData(x.concat, case[0], case[1], 
                            anode.Parents(case[2]), agraph.Args(*case[3]))




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack (of concatenated nodes) --------------------------------- #

PackData = collections.namedtuple("PackData", [
              "pack", "deshelled", "deshell", "concat", 
              "node", "source", "funwrap", 
           ])




def pack_dat(valency=1, layer=0):
 
    deshelled_pack = fake.Packable()
    deshelled_args = fake.ArgsLike(pack=fake.Fun(deshelled_pack))

    if   layer == agraph.minlayer():
         source  = fake.Value()
         node    = anode.Point(source)         
    else:
         source  = fake.NodeLike()
         node    = fake.NodeLike()

    funwrap = fake.Fun(None)
    adxs    = arepeat(fake.Value, valency)
    op      = anode.AdjointOp(funwrap, adxs, source, deshelled_args)
    parents = fake.Parental(next=fake.Fun(node, source, layer, op))  

    concat = fake.Cohesive(
                           layer=fake.Fun(layer),
                           adxs=fake.Fun(adxs),
                           parents=fake.Fun(parents),
                           deshell=fake.Fun(deshelled_args)
                          )
    pack = agraph.Pack(concat)

    return PackData(pack, deshelled_pack, deshelled_args, concat, 
                    node, source, funwrap)
  



# --- Argument envelope ----------------------------------------------------- #

"""

def envelope_dat(nargs=1):

    outval1  = fake.Value()
    argvals1 = arepeat(fake.Value, nargs) 
    args1    = amap(anode.Point, argvals1)

    out1 = fake.NodeLike()
    out2 = fake.NodeLike()
    out3 = fake.NodeLike()

    fun     = fake.Fun(outval1, *argvals1)
    funwrap = fake.Fun(None)

    pack0 = fake.Packable(
                          innermost=fake.Fun(True), 
                         )
    pack1 = fake.Packable(
                          innermost=fake.Fun(False), 
                          deshelled=fake.Fun(pack0),
                          deshell=fake.Fun(args1),
                          fold=fake.Fun(out1, funwrap, outval1)
                         )
    pack2 = fake.Packable(
                          innermost=fake.Fun(False), 
                          deshelled=fake.Fun(pack1),
                          fold=fake.Fun(out2, funwrap, out1)
                         )
    pack3 = fake.Packable(
                          innermost=fake.Fun(False), 
                          deshelled=fake.Fun(pack2),
                          fold=fake.Fun(out3, funwrap, out2)
                         )

    packs    = (pack1, pack2, pack3)
    nodes    = (out1, out2, out3)
    args     = fake.ArgsLike(pack=fake.Fun(pack3))
    envelope = agraph.Envelope(args)
    
    return EnvelopeData(envelope, args, packs, nodes, 
                        outval1, fun, funwrap)



def envelope_dat_001(nargs=1):

    outval1  = fake.Value()
    argvals1 = arepeat(fake.Value, nargs) 
    args1    = amap(anode.Point, argvals1)

    out1 = anode.Point(outval1)  

    fun     = fake.Fun(outval1, *argvals1)
    funwrap = fake.Fun(None)

    pack0 = fake.Packable(
                          innermost=fake.Fun(True), 
                         )
    pack1 = fake.Packable(
                          innermost=fake.Fun(True), 
                          deshelled=fake.Fun(pack0),
                          deshell=fake.Fun(args1),
                          fold=fake.Fun(out1, funwrap, outval1)
                         )

    packs    = (pack1, )
    nodes    = (out1,  )
    args     = fake.ArgsLike(pack=fake.Fun(pack1))
    envelope = agraph.Envelope(args)
    
    return EnvelopeData(envelope, args, packs, nodes, 
                        outval1, fun, funwrap)

"""


EnvelopeData = collections.namedtuple("EnvelopeData", [
                  "envelope", "args", 
                  "packs", "nodes", 
                  "outnode", "outvalue", "fun", "funwrap"
               ])




def envelope_dat(stackdat):

    def make_packs():

        nodes = stackdat.nodes 
        packs = []

        for _nodes in nodes:

           _args  = agraph.Args(*_nodes)
           _packs = _args.pack()

           packs.append(_packs)

        return tuple(reversed(packs))


    packs = make_packs()

    outnode = stackdat.outnodes[-1]
    args    = agraph.Args(*stackdat.nodes[-1])

    envelope = agraph.Envelope(args)

    return EnvelopeData(
                        envelope, args, 
                        packs, stackdat.nodes, 
                        outnode, stackdat.outvalue, 
                        stackdat.fun, stackdat.funwrap
                       )




# --- Node stack ------------------------------------------------------------ #

NodeStackData = collections.namedtuple("NodeStackData", [
                   "outnodes", "outparents", "outvalue",
                   "nodes",    "parents",    "values", 
                   "layers", "adxs", "fun", "funwrap",
                ])




def node_stack_dat_001(gatetype="REVERSE"):

    def gate(parents, op=None):

        if op is None:
           op = fake.Adjoint()

        parents = anode.Parents(parents)

        return {
                "REVERSE": anode.ReverseGate, 
                "FORWARD": anode.ForwardGate,
               }[gatetype](parents, op)


    # --- Input nodes --- #

    values = arepeat(fake.Value, 3)

    layersA = (-1, -1, -1)
    layersB = (-1,  0, -1)
    layersC = (-1,  1,  1)
    layersD = ( 2,  1,  1)

    adxsA = tuple()
    adxsB = (1,)
    adxsC = (1,2)
    adxsD = (0,)

    parentsA = (tuple(),                   tuple(),                   tuple())
    parentsB = (tuple(),                   arepeat(fake.NodeLike, 3), tuple()) 
    parentsC = (parentsB[0],               arepeat(fake.NodeLike, 1), arepeat(fake.NodeLike, 2))
    parentsD = (arepeat(fake.NodeLike, 2), parentsC[1],               parentsC[2])

    nodesA = amap(anode.Point, values)
    nodesB = (
              nodesA[0],  
              anode.Node(nodesA[1], layersB[1], gate(parentsB[1])), 
              nodesA[2], 
             )
    nodesC = (
              nodesB[0],
              anode.Node(nodesB[1], layersC[1], gate(parentsC[1])),          
              anode.Node(nodesB[2], layersC[2], gate(parentsC[2])),       
             )
    nodesD = (
              anode.Node(nodesC[0], layersD[0], gate(parentsD[0])),  
              nodesC[1],
              nodesC[2],
             )

    nodes   = (nodesA,   nodesB,   nodesC,   nodesD)
    parents = (parentsA, parentsB, parentsC, parentsD)
    adxs    = (adxsA,    adxsB,    adxsC,    adxsD)
    layers  = (layersA,  layersB,  layersC,  layersD)


    # --- Output node --- #

    outvalue = fake.Value()

    outparentsA = tuple()
    outparentsB = (nodesB[1],          )
    outparentsC = (nodesC[1], nodesC[2])
    outparentsD = (nodesD[0],          )

    fun     = fake.Fun(outvalue, *values)
    funwrap = fake.Fun(None)

    outnodeA = anode.Point(outvalue)
    outnodeB = anode.Node(outnodeA, 0, gate(outparentsB, anode.AdjointOp(funwrap, adxsB, outnodeA, agraph.Args(*nodesA))))
    outnodeC = anode.Node(outnodeB, 1, gate(outparentsC, anode.AdjointOp(funwrap, adxsC, outnodeB, agraph.Args(*nodesB))))
    outnodeD = anode.Node(outnodeC, 2, gate(outparentsD, anode.AdjointOp(funwrap, adxsD, outnodeC, agraph.Args(*nodesC))))

    outnodes   = (outnodeA,    outnodeB,    outnodeC,    outnodeD)
    outparents = (outparentsA, outparentsB, outparentsC, outparentsD)

    return NodeStackData(
                         outnodes, outparents, outvalue, 
                         nodes, parents, values,
                         layers, adxs,
                         fun, funwrap,
                        )




def node_stack_dat_002(gatetype="REVERSE"):

    # --- Input nodes --- #

    values = arepeat(fake.Value, 3)

    layersA  = (-1, -1, -1)
    adxsA    = tuple()
    parentsA = (tuple(), tuple(), tuple())
    nodesA   = amap(anode.Point, values)

    nodes   = (nodesA,   )
    parents = (parentsA, )
    adxs    = (adxsA,    )
    layers  = (layersA,  )

    # --- Output node --- #

    outvalue    = fake.Value()
    outparentsA = tuple()
    outnodeA    = anode.Point(outvalue)

    outnodes   = (outnodeA,    )
    outparents = (outparentsA, )

    fun     = fake.Fun(outvalue, *values)
    funwrap = fake.Fun(None)

    return NodeStackData(
                         outnodes, outparents, outvalue, 
                         nodes, parents, values,
                         layers, adxs,
                         fun, funwrap,
                        )














