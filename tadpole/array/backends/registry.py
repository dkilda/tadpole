#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import functools
import numpy as np

from tadpole.array.backends.backend import Backend
from tadpole.array.backends.numpy   import NumpyBackend
from tadpole.array.backends.torch   import TorchBackend




###############################################################################
###                                                                         ###
###  Backend registry and its access ports                                  ###
###                                                                         ###
###############################################################################


# --- Unsupported backend error --------------------------------------------- #

class UnsupportedBackendError(Exception):

   def __init__(self, value):
       self.value = value

   def __str__(self):
       return repr(self.value)




# --- Backend registry ------------------------------------------------------ #

class BackendRegistry:

   _backends = {
      "numpy": NumpyBackend, 
      "torch": TorchBackend,
   }

   def __init__(self, default):

       self._default      = default
       self._instantiated = {}
                             

   def create(self, backend):

       if backend in self._instantiated:

          return self._instantiated[backend]

       if backend in type(self)._backends:

          self._instantiated[backend] = type(self)._backends[backend]()
          return self._instantiated[backend]

       raise UnsupportedBackendError(
          f"Backend '{backend}' is not supported."
       )


   def get(self, backend):

       if backend is None:
          backend = self._default

       if isinstance(backend, Backend):
          return backend

       if isinstance(backend, str):
          return self.create(backend)

       raise ValueError((
          f"Invalid backend '{backend}'. " 
          f"The backend input must be a string or a Backend object."
       ))


   def set_default(self, default):

       self._default = default
       return self




# --- A global instance of backend registry and its access ports ------------ #
           
_BACKENDS = BackendRegistry(default="numpy")


def get(backend):

    return _BACKENDS.get(backend)


def get_from(opts):

    return get(opts.pop("backend", None))


def set_default(backend):

    _BACKENDS.set_default(backend)




# --- Extract backend string from input array ------------------------------- #

@functools.lru_cache(None)
def _get_str(cls):

    if issubclass(cls, np.ndarray):
       return "numpy"

    return cls.__module__.split(".")[0]


def get_str(array):

    return _get_str(array.__class__)




