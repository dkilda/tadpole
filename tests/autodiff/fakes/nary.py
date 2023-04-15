#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.autodiff.types as at
import tests.util.fakes       as fake




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy -------------------------------------------------------- #

class ArgProxy(at.ArgProxy):

   def __init__(self, **data):  

       self._data = data


   def _get(self, name, default=None):
       
       return self._data.get(name, default)


   def insert(self, args, x):

       default = fake.Fun(fake.Value())

       return self._get("insert", default)(args, x)


   def extract(self, args):

       default = fake.Fun(fake.Value())

       return self._get("extract", default)(args)




