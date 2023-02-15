#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake

import tadpole.autodiff.node  as tdnode
import tadpole.autodiff.graph as tdgraph
import tadpole.autodiff.grad  as tdgrad




###############################################################################
###                                                                         ###
###  Differential operators: forward and reverse                            ###
###                                                                         ###
###############################################################################


# --- Differential and Graph operators -------------------------------------- #

DiffOpData = collections.namedtuple("DiffOpData", [
                "diffop", "graphop", "root", "fun", 
                "x", "out", 
                "start", "end",
             ])




def diffop(which="REVERSE", x=None, out=None, start=None, end=None):

    if x is None:
       x = fake.Value()

    if out is None:
       out = fake.Value()

    if start is None:
       start = fake.NodeLike(tovalue=fake.Fun(x))

    if end is None:
       end = fake.NodeLike(tovalue=fake.Fun(out))

    fun  = fake.Fun(end, start)
    root = {
            "REVERSE": tdnode.ReverseGate,
            "FORWARD": tdnode.ForwardGate,
           }[which]()

    graphop = tdgrad.GraphOp(root, fun, x)
    diffop  = {
               "REVERSE": tdgrad.ReverseDifferentialOp,
               "FORWARD": tdgrad.ForwardDifferentialOp,
              }[which]()

    return DiffOpData(
                      diffop, graphop, 
                      root, fun, 
                      x, out, start, end 
                     )   




###############################################################################
###                                                                         ###
###  Node network for topological sort and gradient accumulation/summation. ###
###                                                                         ###
###############################################################################


# --- Node network ---------------------------------------------------------- #

NodeNetworkData = collections.namedtuple("NodeNetworkData", [
                    "end", "nodes", "gradmap", "parentmap", "countmap",
                  ])




def forward_node_network(layer=None):

    # --- Nodes and grads --- #

    seed0 = (fake.Value(), )
    seed1 = (fake.Value(), )
    seed2 = (fake.Value(), )

    leaf0_dat = node.reverse_node_dat(0, seed0, seed0, layer)
    leaf1_dat = node.reverse_node_dat(0, seed1, seed1, layer)
    leaf2_dat = node.reverse_node_dat(0, seed2, seed2, layer)

    nodeA_dat = node.reverse_node_dat(
                                      (leaf0_dat.node,       leaf1_dat.node), 
                                      (sum(leaf0_dat.grads), sum(leaf1_dat.grads)),
                                      layer=layer
                                     )
    nodeB_dat = node.reverse_node_dat(
                                      (leaf0_dat.node,       leaf2_dat.node), 
                                      (sum(leaf0_dat.grads), sum(leaf2_dat.grads)),
                                      layer=layer
                                     )
    nodeC_dat = node.reverse_node_dat(
                                      (nodeA_dat.node,       ), 
                                      (sum(nodeA_dat.grads), ),
                                      layer=layer
                                     )
    nodeD_dat = node.reverse_node_dat(
                                      (leaf1_dat.node,       nodeB_dat.node), 
                                      (sum(leaf1_dat.grads), sum(nodeB_dat.grads)), 
                                      layer=layer
                                     )
    nodeE_dat = node.reverse_node_dat(
                                      (nodeC_dat.node,      leaf1_dat.node,      nodeD_dat.node), 
                                      (sum(nodeC_dat.node), sum(leaf1_dat.node), sum(nodeD_dat.node)), 
                                      layer=layer
                                     ) 

    # --- Network data --- #

    node_dats = (
                 nodeE_dat, 
                 nodeD_dat, 
                 nodeB_dat, 
                 leaf2_dat, 
                 nodeC_dat, 
                 nodeA_dat, 
                 leaf1_dat, 
                 leaf0_dat,
                )
  
    end   = nodeE_dat.node
    nodes = tuple(dat.node for dat in node_dats)

    gradmap   = {dat.node: sum(dat.grads) for dat in node_dats}
    parentmap = {dat.node: dat.parents    for dat in node_dats}
    countmap  = {
                 nodeE_dat.node: 0,
                 nodeD_dat.node: 1,
                 nodeB_dat.node: 1,
                 leaf2_dat.node: 1,
                 nodeC_dat.node: 1,
                 nodeA_dat.node: 1,
                 leaf1_dat.node: 3,
                 leaf0_dat.node: 2,
                }

    return NodeNetworkData(end, nodes, gradmap, parentmap, countmap)




def reverse_node_network(layer=None):

    # --- Grads --- #

    gradE = {
             None: fake.Value(), 
             "C": fake.Value(), 
             "1": fake.Value(), 
             "D": fake.Value(),
            } 

    gradD = {
             None: gradE["D"], 
             "1":  fake.Value(), 
             "B":  fake.Value(),
            }
 
    gradB = {
             None: gradD["B"], 
             "0": fake.Value(), 
             "2": fake.Value(),
            }
 
    gradC = {
             None: gradE["C"], 
             "A": fake.Value(),
            }

    gradA = {
             None: gradC["A"], 
             "0": fake.Value(), 
             "1": fake.Value(),
            }
   
    grad0 = {None: gradA["0"] + gradB["0"]}
    grad1 = {None: gradA["1"] + gradD["1"] + gradE["1"]}
    grad2 = {None: gradB["2"]}


    # --- Nodes --- #

    leaf0_dat = node.reverse_node_dat(0, grad0[None], grad0[None], layer)
    leaf1_dat = node.reverse_node_dat(0, grad1[None], grad1[None], layer)
    leaf2_dat = node.reverse_node_dat(0, grad2[None], grad2[None], layer)

    nodeA_dat = node.reverse_node_dat(
                                      (leaf0_dat.node, leaf1_dat.node), 
                                      (gradA["0"], gradA["1"]),
                                      gradA[None],
                                      layer
                                     )
    nodeB_dat = node.reverse_node_dat(
                                      (leaf0_dat.node, leaf2_dat.node), 
                                      (gradB["0"], gradB["2"]),
                                      gradB[None],
                                      layer
                                     )
    nodeC_dat = node.reverse_node_dat(
                                      (nodeA_dat.node,), 
                                      (gradC["A"],),
                                      gradC[None],
                                      layer
                                     )
    nodeD_dat = node.reverse_node_dat(
                                      (leaf1_dat.node, nodeB_dat.node), 
                                      (gradD["1"], gradD["B"]),
                                      gradD[None],
                                      layer
                                     )
    nodeE_dat = node.reverse_node_dat(
                                      (nodeC_dat.node, leaf1_dat.node, nodeD_dat.node), 
                                      (gradE["C"], gradE["1"], gradE["D"]),
                                      gradE[None],
                                      layer
                                     ) 

    # --- Network data --- #

    node_dats = (
                 nodeE_dat, 
                 nodeD_dat, 
                 nodeB_dat, 
                 leaf2_dat, 
                 nodeC_dat, 
                 nodeA_dat, 
                 leaf1_dat, 
                 leaf0_dat,
                )
  
    end   = nodeE_dat.node
    nodes = tuple(dat.node for dat in node_dats)

    gradmap   = {dat.node: dat.seed    for dat in node_dats}
    parentmap = {dat.node: dat.parents for dat in node_dats}
    countmap  = {
                 nodeE_dat.node: 0,
                 nodeD_dat.node: 1,
                 nodeB_dat.node: 1,
                 leaf2_dat.node: 1,
                 nodeC_dat.node: 1,
                 nodeA_dat.node: 1,
                 leaf1_dat.node: 3,
                 leaf0_dat.node: 2,
                }

    return NodeNetworkData(end, nodes, gradmap, parentmap, countmap)




