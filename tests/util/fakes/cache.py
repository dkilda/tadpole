#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import tests.util.fakes as fake



###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Fake class with a cacheable method ------------------------------------ #

class CacheMe:

   def __init__(self):

       self._sentinel = 0


   def sentinel(self):

       return self._sentinel


   def execute(self):

       self._sentinel += 1
       return fake.Value()
