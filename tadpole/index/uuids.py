#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import string
import uuid

import tadpole.util as util




###############################################################################
###                                                                         ###
###  Framework that generates truly unique uuids for tensor indices         ###
###                                                                         ###
###############################################################################


# --- Alphabet product ------------------------------------------------------ #

class AlphabetProduct:

   def __init__(self, alphabet=None):

       if alphabet is None:
          alphabet = string.ascii_uppercase + string.ascii_lowercase

       self._alphabet = alphabet


   def finite(self, repeat):

       return itertools.product(self._alphabet, repeat=repeat)


   def expanding(self, min_repeat):

       return itertools.chain.from_iterable(
          self.finite(repeat) for repeat in itertools.count(min_repeat)
       )


   def expanding_str(self, min_repeat):

       return map("".join, self.expanding(min_repeat))




# --- UUIDs ----------------------------------------------------------------- #

class UUIDs:

   def __init__(self, product, prefix_size, min_repeat):

       self._product     = product
       self._prefix_size = prefix_size
       self._min_repeat  = min_repeat


   @util.cacheable
   def prefix(self):

       return str(uuid.uuid4())[:self._prefix_size]


   @util.cacheable
   def expanding_product(self):

       return self._product.expanding_str(self._min_repeat)


   def next(self, base):

       return f"{base}_{self.prefix()}{next(self.expanding_product())}"




# --- UUID error ------------------------------------------------------------ #

class UUIDError(Exception):

   def __init__(self, value):
       self.value = value

   def __str__(self):
       return repr(self.value)




# --- UUID generator -------------------------------------------------------- #

class UUIDGenerator:

   def __init__(self):

       self._uuids = None 

       
   def setup(self, product=None, prefix_size=6, min_repeat=5): 

       if self._uuids is not None:
          raise UUIDError(
             "Illegal action triggered by uuids.setup(): an attempt to "
             "reset UUIDGenerator. Resetting UUIDGenerator is not permitted."
          )

       if product is None:
          product = AlphabetProduct()

       self._uuids = UUIDs(product, prefix_size, min_repeat)
         

   def next(self, base=None):

       if self._uuids is None:
          self.setup()

       if base is None:
          base = ""

       return self._uuids.next(base)




# --- Create a global UUID generator ---------------------------------------- #

_UUID_GENERATOR = UUIDGenerator()


def setup(self, *args, **kwargs):

    _UUID_GENERATOR.setup(*args, **kwargs)


def next_uuid(base=None):

    return _UUID_GENERATOR.next(base)




