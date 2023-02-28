#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tadpole.util as util




###############################################################################
###                                                                         ###
###  Logical chain: used for equality comparisons and other assertions.     ###
###                                                                         ###
###############################################################################


# --- Helper methods -------------------------------------------------------- #

def refid(x):

    if isinstance(x, (tuple, list)): 
       return tuple(map(id, x))

    return id(x)




# --- Logical chain --------------------------------------------------------- #

class LogicalChain:

   def __init__(self):

       self._chain = []


   def _add(self, cond):

       self._chain.append(cond)
       return self


   def typ(self, x, y):

       return self._add(type(x) == type(y))
 

   def ref(self, x, y):

       return self._add(refid(x) == refid(y))
        

   def val(self, x, y):

       return self._add(x == y)


   def __bool__(self):

       return all(self._chain)


   def __repr__(self):

       rep = ReprChain()
       rep.typ(self)
       return str(rep)




###############################################################################
###                                                                         ###
###  Representation chain: constructs a string representation (repr)        ###
###  of an object.                                                          ###
###                                                                         ###
###############################################################################


# --- Helper methods (for formatting) --------------------------------------- #

def format_type(x):

    return f"{str(type(x))[8:-2]}"


def format_obj(x):

    return f"{format_type(x)}, id = {id(x)}:"


def format_val(name, x):

    return f"{format_type(x)} {name} = {x}"


def format_ref(name, x):

    if isinstance(x, (tuple, list)):
       return format_refs(name, x)

    return f"{format_type(x)} {name} ID = {id(x)}"


def format_refs(name, xs):

    xids = ', '.join(map(lambda v: f"{format_type(v)} ID = {id(v)}", xs))
    return f"{format_type(xs)} {name} : {xids}"


def vjoin(args):

    return '\n  . '.join(filter(None, args))




# --- Representation chain -------------------------------------------------- #

class ReprChain:

   def __init__(self):

       self._str = []


   def _add(self, xstr):

       self._str.append(xstr)
       return self

       
   def typ(self, x):

       return self._add(format_obj(x))

  
   def ref(self, name, x):

       return self._add(format_ref(name, x))
 

   def val(self, name, x):
 
       return self._add(format_val(name, x))   


   def __repr__(self):

       return f"{format_obj(self)}"


   def __str__(self):

       return f"\n{vjoin(self._str)}\n"




