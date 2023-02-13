#!/usr/bin/env python3
# -*- coding: utf-8 -*-




###############################################################################
###                                                                         ###
###  General utility for fakes                                              ###
###                                                                         ###
###############################################################################


# --- NULL constant --------------------------------------------------------- #

class NullType:
   pass

NULL = NullType()




# --- Undefined fake error -------------------------------------------------- #

class UndefinedFakeError(Exception):

   def __init__(self, value):
       self.value = value

   def __str__(self):
       return repr(self.value)




# --- Decorator for fake methods -------------------------------------------- #

def fakeit(fun):

    def wrap(*args, **kwargs):

        msg = f"\nThe behavior of fake method {fun.__name__} is undefined."

        try:
           out = fun(*args, **kwargs)
        except TypeError:
           raise UndefinedFakeError(msg)

        if out is NULL:
           raise UndefinedFakeError(msg)

        return out

    return wrap










