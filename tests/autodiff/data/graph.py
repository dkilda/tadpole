#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.util  as tdutil
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad



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
            "REVERSE": tdnode.ReverseGate,
            "FORWARD": tdnode.ForwardGate,
           }[which]()

    if layer is None:
       layer = 0

    if x is None:
       x = fake.Value()

    if out is None:
       out = fake.Value()

    if start is None:
       start = tdnode.Node(x, layer, root) 

    if end is None:
       src = fake.NodeLike(tovalue=fake.Fun(out))
       end = tdnode.Node(src, layer, fake.GateLike())

    fun     = fake.Fun(end, start)
    graph   = tdgraph.Graph(root)
    graphop = tdgrad.GraphOp(root, fun, x)

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
    make_envelope = fake.Fun(envelope, args)

    fun     = fake.Fun(None)
    funwrap = tdgraph.Differentiable(fun, make_envelope)

    applywrap.update_args(funwrap, fun)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




def nondifferentiable_funwrap_dat(args):

    out = fake.NodeLike()

    envelope      = fake.EnvelopeLike(**{"apply": fake.Fun(out)})
    make_envelope = fake.Fun(envelope, args)

    fun     = fake.Fun(None)
    funwrap = tdgraph.NonDifferentiable(fun, make_envelope)

    return FunWrapData(funwrap, fun, make_envelope, envelope, args, out)




###############################################################################
###                                                                         ###
###  Function arguments and their concatenation.                            ###
###                                                                         ###
###############################################################################


# --- Function arguments ---------------------------------------------------- #

ArgsData = collections.namedtuple("ArgsData", [
              "args", "concat", "pack", 
              "adxs", "nodes", "sources", "layers", "values",
           ])




def args_dat(n=1, adxs=(0,), layers=(0,)):

    info   = []
    layers = list(reversed(layers))

    for i in range(n):

        value = fake.Value()

        if   i in adxs: 
             source = fake.NodeLike(tovalue=fake.Fun(value))
             layer  = layers.pop()
             node   = tdnode.Node(source, layer, fake.GateLike())

        else:
             layer  = tdgraph.minlayer()           
             node   = tdnode.Point(value)
             source = node

        info.append([node, source, layer, value])

    nodes, sources, layers, values = zip(*info)
    args = tdgraph.Args(nodes)

    concat = tdgraph.Concatenation(
                                   tdutil.Sequence(nodes,   n), 
                                   tdutil.Sequence(sources, n), 
                                   tdutil.Sequence(layers,  n)
                                  )
    pack = tdgraph.Pack(concat)

    return ArgsData(args, concat, pack, 
                    adxs, nodes, sources, layers, values)




def nodeargs_dat(n=1, layers=None):

    if layers is None:
       layers = common.arepeat(0, n)

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

        _nodes   = tdutil.Sequence(x.nodes[:size],   size)
        _sources = tdutil.Sequence(x.sources[:size], size)
        _layers  = tdutil.Sequence(x.layers[:size],  size)

        _concat  = tdgraph.Concatenation(_nodes, _sources, _layers)

        concat_history.append(_concat)
         
    nodes   = tdutil.Sequence(x.nodes,   n)
    sources = tdutil.Sequence(x.sources, n)
    layers  = tdutil.Sequence(x.layers,  n)

    concat  = tdgraph.Concatenation(nodes, sources, layers)

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
           -1, tuple(), tuple(), (x.nodes[0], x.nodes[1])
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
                            tdnode.Parents(case[2]), tdgraph.Args(case[3]))




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

    if   layer == tdgraph.minlayer():
         source  = fake.Value()
         node    = tdnode.Point(source)         
    else:
         source  = fake.NodeLike()
         node    = fake.NodeLike()

    funwrap = fake.Fun(None)
    adxs    = common.arepeat(fake.Value, valency)
    op      = tdnode.AdjointOp(funwrap, adxs, source, deshelled_args)
    parents = fake.Parental(next=fake.Fun(node, source, layer, op))  

    concat = fake.Cohesive(
                           layer=fake.Fun(layer),
                           adxs=fake.Fun(adxs),
                           parents=fake.Fun(parents),
                           deshell=fake.Fun(deshelled_args)
                          )
    pack = tdgraph.Pack(concat)

    return PackData(pack, deshelled_pack, deshelled_args, concat, 
                    node, source, funwrap)
  



# --- Argument envelope ----------------------------------------------------- #

EnvelopeData = collections.namedtuple("EnvelopeData", [
                  "envelope", "args", "packs", "nodes", 
                  "value", "fun", "funwrap"
               ])



def envelope_dat(nargs=1):

    outval1  = fake.Value()
    argvals1 = common.arepeat(fake.Value, nargs) 
    args1    = common.amap(tdnode.Point, argvals1)

    out1 = fake.NodeLike(tovalue=fake.Fun(outval1))
    out2 = fake.NodeLike(tovalue=fake.Fun(outval1))
    out3 = fake.NodeLike(tovalue=fake.Fun(outval1))

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
    envelope = tdgraph.Envelope(args)
    
    return EnvelopeData(envelope, args, packs, nodes, 
                        outval1, fun, funwrap)



def envelope_dat_001(nargs=1):

    outval1  = fake.Value()
    argvals1 = common.arepeat(fake.Value, nargs) 
    args1    = common.amap(tdnode.Point, argvals1)

    out1 = tdnode.Point(outval1)  

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
    envelope = tdgraph.Envelope(args)
    
    return EnvelopeData(envelope, args, packs, nodes, 
                        outval1, fun, funwrap)





























