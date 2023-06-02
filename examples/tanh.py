#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

import numpy   as np
import tadpole as td

import timeit
import cProfile


"""
Reproducing autograd's tanh trademark example

"""


'''
PROFILE-1


         25661751 function calls (24096538 primitive calls) in 11.575 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  1300622    0.577    0.000    0.915    0.000 itertools.py:272(__init__)
819448/513046    0.528    0.000    4.190    0.000 cache.py:22(wrap)
   807439    0.466    0.000    1.339    0.000 itertools.py:348(push)
   807439    0.444    0.000    0.745    0.000 itertools.py:293(_apply)
2513720/1361828    0.393    0.000    2.651    0.000 {built-in method builtins.iter}
   451509    0.371    0.000    0.819    0.000 itertools.py:246(apply)
   451509    0.359    0.000    1.364    0.000 itertools.py:284(_list)
  2277288    0.332    0.000    0.567    0.000 {built-in method builtins.isinstance}
   720363    0.320    0.000    2.408    0.000 itertools.py:323(__iter__)
   244127    0.317    0.000    1.614    0.000 graph.py:364(attach)
241438/154698    0.288    0.000    0.452    0.000 node.py:473(_node_by_type)
   390638    0.275    0.000    0.823    0.000 graph.py:323(__init__)
   823564    0.270    0.000    0.270    0.000 {built-in method builtins.getattr}
   146511    0.233    0.000    2.704    0.000 graph.py:308(train_args)
    72235    0.222    0.000    2.677    0.000 graph.py:291(denodify)
   705133    0.186    0.000    0.259    0.000 itertools.py:391(execute)
   451509    0.178    0.000    0.190    0.000 itertools.py:257(restore)
   493183    0.176    0.000    0.210    0.000 itertools.py:232(__init__)
   390638    0.163    0.000    1.130    0.000 graph.py:274(_nodify)
   146511    0.157    0.000    1.288    0.000 graph.py:272(nodify)
   154698    0.155    0.000    0.607    0.000 node.py:471(_create)
    65771    0.149    0.000    1.356    0.000 graph.py:480(deshell)
   343628    0.145    0.000    0.979    0.000 graph.py:446(_layers)
   148183    0.133    0.000    0.721    0.000 node.py:513(point)
    72235    0.133    0.000    3.562    0.000 graph.py:243(_denodified)
   294663    0.133    0.000    1.100    0.000 graph.py:457(layer)
   807439    0.129    0.000    0.129    0.000 itertools.py:386(__init__)
   466930    0.128    0.000    0.128    0.000 {built-in method _abc._abc_instancecheck}
    29788    0.120    0.000    7.811    0.000 graph.py:651(apply)
   200030    0.119    0.000    1.033    0.000 graph.py:452(innermost)
   244127    0.111    0.000    1.725    0.000 node.py:432(concat)
   466930    0.107    0.000    0.235    0.000 abc.py:96(__instancecheck__)
   451509    0.103    0.000    0.185    0.000 itertools.py:241(__iter__)
    74276    0.095    0.000    2.247    0.000 graph.py:248(concat)
    48965    0.094    0.000    0.557    0.000 graph.py:462(adxs)
   415863    0.090    0.000    1.170    0.000 graph.py:382(layers)

'''




'''
PROFILE-2

         25701622 function calls (24135755 primitive calls) in 11.439 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   11.439   11.439 {built-in method builtins.exec}
        1    0.000    0.000   11.439   11.439 <string>:1(<module>)
        1    0.000    0.000   11.439   11.439 tanh.py:101(main)
     21/6    0.000    0.000   11.350    1.892 nary.py:107(__call__)
     21/6    0.010    0.000   11.350    1.892 nary.py:60(__call__)
     21/6    0.000    0.000   11.342    1.890 grad.py:34(gradient)
       21    0.000    0.000   11.276    0.537 grad.py:122(grad)
       21    0.005    0.000   11.276    0.537 grad.py:222(grads)
     6340    0.004    0.000   11.063    0.002 node.py:447(grads)
     6340    0.016    0.000   11.059    0.002 node.py:358(grads)
     6340    0.022    0.000   10.412    0.002 grad.py:586(add)
    13812    0.011    0.000    8.259    0.001 adjointmap.py:139(<genexpr>)
     7493    0.009    0.000    8.248    0.001 adjointmap.py:88(adjfun)
    29788    0.121    0.000    7.607    0.000 graph.py:651(apply)
     21/6    0.000    0.000    6.050    1.008 grad.py:65(diffop_reverse)
     21/6    0.000    0.000    6.049    1.008 grad.py:282(execute)
     21/6    0.000    0.000    6.049    1.008 graph.py:91(build)
     21/6    0.000    0.000    6.049    1.008 graph.py:33(wrap)
     21/6    0.000    0.000    6.049    1.008 nary.py:109(unary_fun)
    10427    0.018    0.000    5.849    0.001 graph.py:130(__call__)
    10427    0.076    0.000    5.791    0.001 graph.py:659(applywrap)
     2281    0.003    0.000    4.842    0.002 interaction.py:169(match)
    19361    0.049    0.000    4.419    0.000 graph.py:168(__call__)
819448/513046    0.532    0.000    4.207    0.000 cache.py:22(wrap)
    72235    0.130    0.000    3.493    0.000 graph.py:243(_denodified)
     2281    0.008    0.000    3.412    0.001 interaction.py:202(reshape_like)
2514431/1362539    0.423    0.000    2.667    0.000 {built-in method builtins.iter}
    72235    0.217    0.000    2.644    0.000 graph.py:291(denodify)
   146511    0.225    0.000    2.634    0.000 graph.py:308(train_args)

'''




x = td.TensorGen(np.linspace(-7, 7, 200), (td.IndexLit("i", 200),))


def tanh(x):
    return (1.0 - td.exp(-x)) / (1.0 + td.exp(-x)) 


def grad(fun):
    return td.gradient(fun)


def deriv(fun):
    return td.derivative(fun)


def plot(*args):
    plt.plot(*map(td.asdata, args))


def main():
    plot(
         x, tanh(x), 
         x, grad(tanh)(x),
         x, grad(grad(tanh))(x), 
         x, grad(grad(grad(tanh)))(x), 
         x, grad(grad(grad(grad(tanh))))(x), 
         x, grad(grad(grad(grad(grad(tanh)))))(x), 
         x, grad(grad(grad(grad(grad(grad(tanh))))))(x), 
        )
    

cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

cProfile.run('main()', sort='tottime')


plt.savefig("tanh_reverse.png")


    
"""
plot(
     x, tanh(x), 
     x, grad(tanh)(x),
     x, grad(grad(tanh))(x), 
     x, grad(grad(grad(tanh)))(x), 
     x, grad(grad(grad(grad(tanh))))(x), 
     x, grad(grad(grad(grad(grad(tanh)))))(x), 
     x, grad(grad(grad(grad(grad(grad(tanh))))))(x), 
    )
"""

"""

plot(
     x, tanh(x), 
     x, deriv(tanh)(x),
     x, deriv(deriv(tanh))(x), 
     x, deriv(deriv(deriv(tanh)))(x), 
     x, deriv(deriv(deriv(deriv(tanh))))(x), 
     x, deriv(deriv(deriv(deriv(deriv(tanh)))))(x), 
     x, deriv(deriv(deriv(deriv(deriv(deriv(tanh))))))(x), 
    )
plt.savefig("tanh_forward.png")

"""




