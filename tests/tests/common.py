#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.testing as nptest




class NodeEq:

   def __init__(self, nodeA, nodeB):

       self._nodeA = nodeA
       self._nodeB = nodeB


   def __bool__(self):

       id(self._nodeA._nodule) == id(self._nodeB._nodule)





def assert_point(this, other):
 
    return assert_nodule(this, other)



class ComparableNode:

   def __init__(self, node):

       self._node = node


   def 




def ids(xs):

    return tuple(map(id, xs))



def assert_simple_node(this, other):

    assert type(this) == type(other)
 
    assert id(this._source) == id(other._source)
    assert this._layer      == other._layer



def assert_composite_node(this, other):
 
    assert type(this) == type(other)

    assert this._nodule == other._nodule
    assert this._gate   == other._gate



def assert_gate(this, other):

    assert type(this) == type(other)

    assert this._parents == other._parents
    assert this._fun     == other._fun



def assert_logic(this, other):

    assert type(this) == type(other)

    assert this._parents == other._parents
    assert this._adxs    == other._adxs

    assert id(this._out)   == id(other._out)
    assert ids(this._args) == ids(other._args)





# assert simple_node_eq(out, ans)





def value_eq(this, other, rtol=2**(-16), atol=2**(-32)):

    return np.allclose(this, other, rtol=rtol, atol=atol)





class Equals:

   def __init__(self, this, other, evaluate):

       self._this     = this
       self._other    = other
       self._evaluate = evaluate


   def __bool__(self):

       return self._evaluate(self._this, self._other)




class SimpleNodeEq:

   def __init__(self, this, other):

       self._eq = Equals(this, other, )







def make_bool(cls):

    @functools.wraps(cls, updated=())
    class Derived(cls):

       def __init__(self, x, y):

           self._x = x
           self._y = y


       def __bool__(self):

           return self._evaluate(self._x, self._y)


    return Derived



@make_bool
class SimpleNodeEq:

   def _evaluate(self, x, y):

       return all((
                   type(x)       == type(y),
                   id(x._source) == id(y._source),
                   x._layer      == y._layer,
                 ))





# --- Equality comparisons -------------------------------------------------- #

def point_eq(x, y):

    return all((
                type(x)       == type(y),
                id(x._source) == id(y._source),
                x._layer      == y._layer,
              ))



def nodule_eq(x, y):

    return all((
                type(x)   == type(y),
                x._source == y._source,
                x._layer  == y._layer,
              ))




def node_eq(x, y):

    return all((
                type(x)   == type(y), 
                x._nodule == y._nodule,
                x._gate   == y._gate,               
              ))




def gate_eq(x, y):

    return all((
                type(x)    == type(y),
                x._parents == y._parents,
                x._fun     == y._fun,
              ))




def logic_eq(x, y):

    return all((
                type(x)      == type(y),
                x._parents   == y._parents,
                x._adxs      == y._adxs,
                id(x._out)   == id(y._out),
                ids(x._args) == ids(y._args),
              ))









def node_signature(x):

    ComparableNode(x)



def assert_node(this, other):

    this




    assert id(this._source) == id(other._source)
    assert this._layer      == other._layer
    
        





# --- Assertion for array data ---------------------------------------------- #

def assert_close(x, ans, rtol=2**(-16), atol=2**(-32)):

    nptest.assert_allclose(x, ans, rtol=rtol, atol=atol) 






























































