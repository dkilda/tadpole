#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import itertools
import numpy as np

import tests.array.fakes as fake
import tests.array.data  as data

import tadpole.util           as util
import tadpole.array.backends as backends
import tadpole.array.core     as core
import tadpole.array.function as function




###############################################################################
###                                                                         ###
###  General framework for array function calls.                            ###
###                                                                         ###
###############################################################################


# --- Content --------------------------------------------------------------- #

class TestContent:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_attach(self, backend, nargs):

       w       = data.function_dat(backend, nargs)
       content = function.Content()

       for warray, wdata in w.arrays_and_datas:
           content = content.attach(warray, wdata)

       assert content == w.content
           

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_iter(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert tuple(w.content) == tuple(w.arrays_and_datas)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_len(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert len(w.content) == nargs




# --- Visit ----------------------------------------------------------------- #

class TestVisit:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_execute(self, backend, nargs):

       w = data.visit_dat(backend, nargs)
       assert w.funcall.execute() == w.out 




# --- Function call --------------------------------------------------------- #

class TestFunCall:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_execute(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert w.funcall.execute() == w.out 




# --- Args ------------------------------------------------------------------ #

class TestArgs:

   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_pluginto(self, backend, nargs):

       w = data.function_dat(backend, nargs)

       funcall = function.FunCall(w.fun)
       args    = function.Args(*w.arrays)

       assert w.args.pluginto(funcall) == w.out


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_iter(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert tuple(w.args) == tuple(w.arrays)


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_reversed(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert tuple(reversed(w.args)) == tuple(reversed(w.arrays))


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_len(self, backend, nargs):

       w = data.function_dat(backend, nargs)
       assert len(w.args) == nargs


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_contains(self, backend, nargs):

       w = data.function_dat(backend, nargs)

       for x in w.arrays:
           assert x in w.args


   @pytest.mark.parametrize("backend", ["numpy"])
   @pytest.mark.parametrize("nargs",   [2])
   def test_getitem(self, backend, nargs):

       w = data.function_dat(backend, nargs)

       for i in range(nargs):
           assert w.args[i] == w.arrays[i]




