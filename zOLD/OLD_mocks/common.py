#!/usr/bin/env python3
# -*- coding: utf-8 -*-



###############################################################################
###                                                                         ###
###  General utility for mocks                                              ###
###                                                                         ###
###############################################################################


# --- NULL constant --------------------------------------------------------- #

class NullType:
   pass

NULL = NullType()




# --- Undefined mock error -------------------------------------------------- #

class UndefinedMockError(Exception):

   def __init__(self, value):
       self.value = value

   def __str__(self):
       return repr(self.value)




# --- Decorator for mock methods -------------------------------------------- #

def mockify(fun):

    def wrap(*args, **kwargs):

        msg = f"\nThe behavior of mock method {fun.__name__} is undefined."

        try:
           out = fun(*args, **kwargs)
        except TypeError:
           raise UndefinedMockError(msg)

        if out is NULL:
           raise UndefinedMockError(msg)

        return out

    return wrap




