#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import collections
from tests.common import arepeat, arange, amap

import tests.autodiff.fakes as fake
import tests.autodiff.data  as data

import tadpole.util           as util
import tadpole.autodiff.types as at
import tadpole.autodiff.node  as an
import tadpole.autodiff.graph as ag
import tadpole.autodiff.grad  as ad




###############################################################################
###                                                                         ###
###  Topological sort of the computation graph                              ###
###                                                                         ###
###############################################################################


# --- Child count ----------------------------------------------------------- #

ChildCountData = collections.namedtuple("ChildCountData", [
                    "count",     "count1",     "count2",
                    "parentmap", "parentmap1", "parentmap2",
                    "countmap",  "countmap1",  "countmap2",
                    "node",      "node1", 
                    "parents",   "parents1",
                 ])




def childcount_dat(valency=1, **countmaps):

    dat  = data.reverse_node_dat(valency)
    dat1 = data.reverse_node_dat(valency)
    
    parentmap  = {}
    parentmap1 = {dat.node: dat.parents}
    parentmap2 = {dat.node: dat.parents, dat1.node: dat1.parents}   

    countmap  = countmaps.get("countmap",  {})
    countmap1 = countmaps.get("countmap1", {})
    countmap2 = countmaps.get("countmap2", {})

    count  = ad.ChildCount(parentmap,  countmap)
    count1 = ad.ChildCount(parentmap1, countmap1)
    count2 = ad.ChildCount(parentmap2, countmap2) 

    return ChildCountData(count,       count1,       count2,     
                          parentmap,   parentmap1,   parentmap2, 
                          countmap,    countmap1,    countmap2,
                          dat.node,    dat1.node, 
                          dat.parents, dat1.parents)




###############################################################################
###                                                                         ###
###  Node network for topological sort and gradient accumulation/summation. ###
###                                                                         ###
###############################################################################


# --- Node network ---------------------------------------------------------- #

NodeNetworkData = collections.namedtuple("NodeNetworkData", [
                     "end", "start", "nodes", "leaves", 
                     "gradmap", "parentmap", "countmap",
                  ])




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

    return NodeNetworkData(end, leaves[0], nodes, leaves, 
                           gradmap, parentmap, countmap)




def reverse_node_network_dat(layer=None):
 
    # --- Grads --- #

    gradE = {
             None: fake.Value(),
             "C":  fake.Value(),
             "1":  fake.Value(),
             "D":  fake.Value(),
            } 

    gradD = {
             None: gradE["D"], 
             "1":  fake.Value(), 
             "B":  fake.Value(),
            }
 
    gradB = {
             None: gradD["B"], 
             "0":  fake.Value(), 
             "2":  fake.Value(),
            }
 
    gradC = {
             None: gradE["C"], 
             "A":  fake.Value(),
            }

    gradA = {
             None: gradC["A"], 
             "0":  fake.Value(), 
             "1":  fake.Value(),
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

    return NodeNetworkData(end, leaves[0], nodes, leaves,
                           gradmap, parentmap, countmap)




