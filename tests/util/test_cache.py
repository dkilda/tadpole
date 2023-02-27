#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

import tests.common     as common
import tests.util.fakes as fake
import tests.util.data  as data

import tadpole.util as util




###############################################################################
###                                                                         ###
###  Cache for methods with one-time evaluation                             ###
###                                                                         ###
###############################################################################


# --- Decorator for cacheable methods --------------------------------------- #

class TestCacheable:

   def test_cacheable(self):

       class CacheMe(fake.CacheMe):

          @util.cacheable
          def execute(self):
              return super().execute()

       x   = CacheMe()
       ans = x.execute()

       assert x.execute()  == ans
       assert x.sentinel() == 1


   def test_cacheable_001(self):

       class CacheMe(fake.CacheMe):
          pass

       x   = CacheMe()
       ans = x.execute()

       assert x.execute()  != ans
       assert x.sentinel() == 2


   def test_cacheable_002(self):

       class CacheMe(fake.CacheMe):

          @util.cacheable
          def execute(self):
              return super().execute()

       x = CacheMe()
       assert x.sentinel() == 0


   def test_cacheable_003(self):

       class CacheMe(fake.CacheMe):

          @util.cacheable
          def execute(self):
              return super().execute()

       x1   = CacheMe()
       ans1 = x1.execute()

       x2   = CacheMe()
       ans2 = x2.execute()

       assert x1.execute()  == ans1
       assert x2.execute()  == ans2
       assert x1.sentinel() == 1
       assert x2.sentinel() == 1




