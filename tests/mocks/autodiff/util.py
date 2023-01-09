#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Cache for methods with one-time evaluation ---------------------------- # 

class MockCacheFun:

   def __init__(self, call=NULL):

       self._call = call


   @mockify
   def __call__(self, fun, *args, **kwargs):

       return self._call[fun]




###############################################################################
###                                                                         ###
###  A quasi-immutable stack data structure                                 ###
###                                                                         ###
###############################################################################


# --- Stack ----------------------------------------------------------------- #

class MockStack:

   def __init__(self, lst=NULL):  

       self._lst = lst


   @mockify
   def push(self, x):

       return self.__class__(self._lst.append(x))


   @mockify
   def pop(self):

       return self.__class__(self._lst[:-1])


   @mockify
   def top(self):

       return self._lst[-1]


   @mockify
   def tolist(self):

       return self._lst


   @mockify
   def iter(self):

       return reversed(iter(self._lst))


   @mockify
   def riter(self):

       return iter(self._lst)


   @mockify
   def size(self):

       return len(self._lst)


   @mockify
   def empty(self):

       return self.size() == 0




###############################################################################
###                                                                         ###
###  Argument proxy: represents a variable in an argument list at a given   ###
###                  argument index. Performs insertion and extraction of   ###
###                  this variable to/from the argument list.               ###
###                                                                         ###
###############################################################################


# --- Argument proxy -------------------------------------------------------- #

class MockArgProxy(ArgProxy):

   def __init__(self, insert=NULL, extract=NULL):

       self._insert  = insert
       self._extract = extract


   @mockify
   def insert(self, args, x):
 
       return self._insert


   @mockify
   def extract(self, args):

       return self._extract




































