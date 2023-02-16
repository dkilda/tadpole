#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections

import tests.common         as common
import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

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
                "diffop", "graphop", 
                "graph", "layer", "root", "fun", 
                "x", "out", "start", "end",
             ])




def diffop_dat(which="REVERSE", layer=None, 
               x=None, out=None, start=None, end=None):

    graph_dat = data.graph_dat(which, layer, x, out, start, end)

    graphop = tdgrad.GraphOp(graph_dat.root, graph_dat.fun, graph_dat.x)
    diffop  = {
               "REVERSE": tdgrad.ReverseDifferentialOp,
               "FORWARD": tdgrad.ForwardDifferentialOp,
              }[which](graph_dat.fun, graph_dat.x)

    return DiffOpData(diffop, *graph_dat)   




###############################################################################
###                                                                         ###
###  Node network for topological sort and gradient accumulation/summation. ###
###                                                                         ###
###############################################################################


# --- Node network ---------------------------------------------------------- #

NodeNetworkData = collections.namedtuple("NodeNetworkData", [
                     "end", "nodes", "leaves", 
                     "gradmap", "parentmap", "countmap",
                  ])





def forward_node_network_dat_001(layer=None):

    # --- Grads --- #

    seed = (fake.Value(), )

    grad0 = {"seed": seed, "grads": seed}
    grad1 = {"seed": seed, "grads": seed}
    grad2 = {"seed": seed, "grads": seed}

    gradA = {
             "seed":  (sum(grad0["grads"]), sum(grad1["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }
    gradB = {
             "seed":  (sum(grad0["grads"]), sum(grad2["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }
    gradC = {
             "seed":  (sum(gradA["grads"]), ),
             "grads": (fake.Value(),        ),
            }
    gradD = {
             "seed":  (sum(grad1["grads"]), sum(gradB["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }
    gradE = {
             "seed":  (sum(gradC["grads"]), sum(grad1["grads"]), sum(gradD["grads"])),
             "grads": (fake.Value(),        fake.Value(),        fake.Value()),
            }


    # --- Nodes --- #

    leaf0_dat = data.forward_node_dat(0, grad0["grads"], grad0["seed"], layer)
    leaf1_dat = data.forward_node_dat(0, grad1["grads"], grad1["seed"], layer)
    leaf2_dat = data.forward_node_dat(0, grad2["grads"], grad2["seed"], layer)

    nodeA_dat = data.forward_node_dat(
                                      (leaf0_dat.node, leaf1_dat.node), 
                                      gradA["grads"], 
                                      gradA["seed"], 
                                      layer
                                     )
    nodeB_dat = data.forward_node_dat(
                                      (leaf0_dat.node, leaf2_dat.node), 
                                      gradB["grads"], 
                                      gradB["seed"],
                                      layer
                                     )
    nodeC_dat = data.forward_node_dat(
                                      (nodeA_dat.node, ), 
                                      gradC["grads"], 
                                      gradC["seed"],
                                      layer
                                     )
    nodeD_dat = data.forward_node_dat(
                                      (leaf1_dat.node, nodeB_dat.node), 
                                      gradD["grads"], 
                                      gradD["seed"],
                                      layer
                                     )
    nodeE_dat = data.forward_node_dat(
                                      (nodeC_dat.node, leaf1_dat.node, nodeD_dat.node), 
                                      gradE["grads"], 
                                      gradE["seed"], 
                                      layer
                                     ) 

    # --- Network data --- #

    node_dats = (
                 nodeE_dat, 
                 nodeD_dat, # OK
                 nodeB_dat, 
                 leaf2_dat,
                 nodeC_dat, # OK
                 nodeA_dat, 
                 leaf1_dat, 
                 leaf0_dat,
                )
  
    end    = nodeE_dat.node
    nodes  = tuple(dat.node for dat in node_dats) 
    leaves = (leaf0_dat.node, leaf1_dat.node, leaf2_dat.node)

    gradmap   = {dat.node: sum(dat.grads) for dat in node_dats}
    parentmap = {dat.node: dat.parents    for dat in node_dats}
    countmap  = {
                 nodeA_dat.node: 0,
                 leaf0_dat.node: 1, 
                }

    return NodeNetworkData(end, nodes, leaves, 
                           gradmap, parentmap, countmap)





def forward_node_network_dat(layer=None):

    # --- Grads --- #

    seed = (fake.Value(), )

    grad0 = {"seed": seed, "grads": seed}
    grad1 = {"seed": seed, "grads": seed}
    grad2 = {"seed": seed, "grads": seed}

    gradA = {
             "seed":  (sum(grad0["grads"]), sum(grad1["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }

    gradB = {
             "seed":  (sum(grad0["grads"]), sum(grad2["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }

    gradC = {
             "seed":  (sum(gradA["grads"]), ),
             "grads": (fake.Value(),        ),
            }

    gradD = {
             "seed":  (sum(grad1["grads"]), sum(gradB["grads"])),
             "grads": (fake.Value(),        fake.Value()),
            }

    gradE = {
             "seed":  (sum(gradC["grads"]), sum(grad1["grads"]), sum(gradD["grads"])),
             "grads": (fake.Value(),        fake.Value(),        fake.Value()),
            }


    # --- Nodes --- #

    leaf0_dat = data.forward_node_dat(0, grad0["grads"], grad0["seed"], layer)
    leaf1_dat = data.forward_node_dat(0, grad1["grads"], grad1["seed"], layer) 
    leaf2_dat = data.forward_node_dat(0, grad2["grads"], grad2["seed"], layer) 

    nodeA_dat = data.forward_node_dat(
                                      (leaf0_dat.node, leaf1_dat.node), 
                                      gradA["grads"], 
                                      gradA["seed"], 
                                      layer
                                     )
    nodeB_dat = data.forward_node_dat(
                                      (leaf0_dat.node, leaf2_dat.node), 
                                      gradB["grads"], 
                                      gradB["seed"],
                                      layer
                                     )
    nodeC_dat = data.forward_node_dat(
                                      (nodeA_dat.node, ), 
                                      gradC["grads"], 
                                      gradC["seed"],
                                      layer
                                     )
    nodeD_dat = data.forward_node_dat(
                                      (leaf1_dat.node, nodeB_dat.node), 
                                      gradD["grads"], 
                                      gradD["seed"],
                                      layer
                                     )
    nodeE_dat = data.forward_node_dat(
                                      (nodeC_dat.node, leaf1_dat.node, nodeD_dat.node), 
                                      gradE["grads"], 
                                      gradE["seed"], 
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
  
    end    = nodeE_dat.node
    nodes  = tuple(dat.node for dat in node_dats) 
    leaves = (leaf0_dat.node, leaf1_dat.node, leaf2_dat.node)

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

    return NodeNetworkData(end, nodes, leaves, 
                           gradmap, parentmap, countmap)




def reverse_node_network_dat(layer=None):

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

    leaf0_dat = data.reverse_node_dat(0, grad0[None], grad0[None], layer)
    leaf1_dat = data.reverse_node_dat(0, grad1[None], grad1[None], layer)
    leaf2_dat = data.reverse_node_dat(0, grad2[None], grad2[None], layer)

    nodeA_dat = data.reverse_node_dat(
                                      (leaf0_dat.node, leaf1_dat.node), 
                                      (gradA["0"], gradA["1"]),
                                      gradA[None],
                                      layer
                                     )
    nodeB_dat = data.reverse_node_dat(
                                      (leaf0_dat.node, leaf2_dat.node), 
                                      (gradB["0"], gradB["2"]),
                                      gradB[None],
                                      layer
                                     )
    nodeC_dat = data.reverse_node_dat(
                                      (nodeA_dat.node,), 
                                      (gradC["A"],),
                                      gradC[None],
                                      layer
                                     )
    nodeD_dat = data.reverse_node_dat(
                                      (leaf1_dat.node, nodeB_dat.node), 
                                      (gradD["1"], gradD["B"]),
                                      gradD[None],
                                      layer
                                     )
    nodeE_dat = data.reverse_node_dat(
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
  
    end    = nodeE_dat.node
    nodes  = tuple(dat.node for dat in node_dats)
    leaves = (leaf0_dat.node, leaf1_dat.node, leaf2_dat.node)

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

    return NodeNetworkData(end, nodes, leaves,
                           gradmap, parentmap, countmap)




