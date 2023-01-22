#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.node as tdnode

from tests.mocks.common import NULL, mockify



###############################################################################
###                                                                         ###
###  Node glue: code for glueing the input nodes together                   ###
###                                                                         ###
###############################################################################


# --- Node train ------------------------------------------------------------ #

class NodeTrain:

   def __init__(self, with_node=NULL, with_meta=NULL, concatenate=NULL):

       self._with_node   = with_node
       self._with_meta   = with_meta
       self._concatenate = concatenate 


   @mockify
   def with_node(self, node):

       return self._with_node


   @mockify
   def with_meta(self, source, layer):

       return self._with_node


   @mockify
   def concatenate(self):

       return self._concatenate 






























































