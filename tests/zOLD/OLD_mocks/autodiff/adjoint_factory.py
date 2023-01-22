#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tests.mocks.common import mockify, NULL




###############################################################################
###                                                                         ###
###  JVP factory                                                            ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) JVP function --------------------------------------- #

class MockNetJvpFun:

   def __init__(self, call=NULL):
 
       self._call = call


   @mockify
   def __call__(self, gs, adxs, out, *args):

       return self._call




# --- JVP factory ----------------------------------------------------------- #

class MockJvpFactory:

   def __init__(self, jvp=NULL):

       self._jvp = jvp


   @mockify
   def jvp(self, parent_gs, adxs, out, *args):

       return self._jvp


   @classmethod
   def add(cls, fun, *jvpfuns, adxs=None):

       return cls


   @classmethod
   def add_combo(cls, fun, jvpfun):

       return cls




###############################################################################
###                                                                         ###
###  VJP factory                                                            ###
###                                                                         ###
###############################################################################


# --- Net (concatenated) VJP function --------------------------------------- #

class MockNetVjpFun:

   def __init__(self, call=NULL):
 
       self._call = call


   @mockify
   def __call__(self, adxs, out, *args):

       return self._call




# --- VJP factory ----------------------------------------------------------- #

class MockVjpFactory:

   def __init__(self, vjp=NULL):

       self._vjp = vjp


   @mockify
   def vjp(self, adxs, out, *args):

       return self._vjp


   @classmethod
   def add(cls, fun, *vjpfuns, adxs=None):

       return cls


   @classmethod
   def add_combo(cls, fun, vjpfun):

       return cls






