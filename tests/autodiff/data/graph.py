#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.util  as tdutil
import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph




###############################################################################
###                                                                         ###
###  Autodiff computation graph                                             ###
###                                                                         ###
###############################################################################


# --- Graph ----------------------------------------------------------------- #

GraphData = collections.namedtuple("GraphData", [
               "graph", "layer", "root", "fun",
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

    fun   = fake.Fun(end, start)
    graph = tdgraph.Graph(root)

    return GraphData(graph, layer, root, fun, x, out, start, end)




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
              "args", "adxs", "nodes", "sources", "layers", "values"
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

    return ArgsData(args, adxs, nodes, sources, layers, values)




def nodeargs_dat(n=1, layers=None):

    if layers is None:
       layers = common.arepeat(0, n)

    return args_dat(n, range(n), layers)



      
# --- Concatenation of nodes ------------------------------------------------ #

ConcatData = collections.namedtuple("ConcatData", [
                       "concat", "nodes", "sources", "layers"
                    ])



def concat_dat(nodes, sources, layers):

    n = len(nodes)

    nodes   = tdutil.Sequence(nodes,   n)
    sources = tdutil.Sequence(sources, n)
    layers  = tdutil.Sequence(layers,  n)

    concat = tdgraph.Concatenation(nodes, sources, layers)

    return ConcatData(concat, nodes, sources, layers)




def concat_args_dat(args_dat):

    return concat_dat(
                      args_dat.nodes, 
                      args_dat.sources, 
                      args_dat.layers
                     )




###############################################################################
###                                                                         ###
###  Argument pack and envelope, which enable us to operate on all          ###
###  arguments as one unit.                                                 ###
###                                                                         ###
###############################################################################


# --- Argument pack and envelope -------------------------------------------- #

PackData = collections.namedtuple("PackData", [
              "envelope", "pack", "concat", "args"
           ])




def pack_dat(args, concat):

    pack       = tdgraph.Pack(concat)
    envelope   = tdgraph.Envelope(args)

    return PackData(envelope, pack, concat, args)




def pack_args_dat(args_dat):

    concat_dat = concat_args_dat(args_dat)

    return pack_dat(args_dat.args, concat_dat.concat)





