#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import collections
import string
import uuid




# --- UUID generator class -------------------------------------------------- #

class UUIDGenerator:

   _alphabet_str = string.ascii_uppercase + string.ascii_lowercase

   def __init__(self, rand_prefix_len=6):

       self._uuid_generator_fun = None
       self._rand_prefix_str    = str(uuid.uuid4())[:rand_prefix_len]


   # --- Intra-class access to the alphabet and random prefix strings --- #

   @property
   def _alphabet(self):
       return UUIDGenerator._alphabet_str

   @property
   def _rand_prefix(self):
       return self._rand_prefix_str


   # --- Implementation of the UUID generator --- #

   def _alphabet_product(self, repeat):
       return itertools.product(self._alphabet, repeat=repeat)


   def _sequence_of_alphabet_products(self, min_reps):
       return itertools.chain.from_iterable(
          self._alphabet_product(repeat) 
             for repeat in itertools.count(min_reps))


   def _sequence_of_alphabet_product_strings(self, min_reps):
       return map("".join, self._sequence_of_alphabet_products(min_reps))


   def _uuid_generator(self, min_reps):

       if self._uuid_generator_fun is None:
          self._uuid_generator_fun = \
             self._sequence_of_alphabet_product_strings(min_reps)

       return self._uuid_generator_fun 

 
   # --- Functor call to generate UUID's --- #

   def __call__(self, base="", min_reps=5):
       return f"{base}_{self._rand_prefix}{next(self._uuid_generator(min_reps))}"




# --- Create a global UUID generator ---------------------------------------- #

UUID_GENERATOR = UUIDGenerator()

def new_uuid(base=""):
    return UUID_GENERATOR(base)

















































