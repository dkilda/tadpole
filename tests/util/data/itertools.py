#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import tadpole.util as util




# --- Loop iterator --------------------------------------------------------- #

LoopData = collections.namedtuple("LoopData", [
              "loop", "xs", "reversed_xs", "last", 
              "first", "next", "stop",
           ]) 




def loop_dat(): 

    xs  = (1,4,7,10,13,16,19,22)
    rxs = (13,10,7,4,1) 

    first_ = 1
    last_  = 13

    def next_(x):
        return x+3

    def stop_(x):
        return x > 15 

    loop = util.Loop(first_, next_, stop_)

    return LoopData(loop, xs, rxs, last_, first_, next_, stop_)








