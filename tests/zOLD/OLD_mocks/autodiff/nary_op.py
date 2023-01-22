#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.mocks.common import mockify, NULL




###############################################################################
###                                                                         ###
###  Nary operator: decorator that converts unary operators into nary ones  ###
###                                                                         ###
###############################################################################


# --- Nary operator --------------------------------------------------------- #

class MockNaryOp:

   def __init__(self, call=NULL):

       self._call = call


   @mockify
   def __call__(self, *args, **kwargs):

       return self._call



















































