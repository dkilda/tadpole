#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.backends as backends
import tadpole.array.util     as util




# --- Gradient interface ---------------------------------------------------- #

class Grad(abc.ABC):

   @abc.abstractmethod
   def __iadd__(self, other):
       pass

   @abc.abstractmethod
   def __add__(self, other):
       pass

   @abc.abstractmethod
   def __mul__(self, other):
       pass



# --- Dense gradient -------------------------------------------------------- #

class DenseGrad(Grad): # FIXME so far it seems identical to Array...

   def __init__(self, array):

       self._array = array


   def __neg__(self):

       return self.__class__(ops.neg(self._array))


   def __add__(self, other):
     
       if isinstance(other, self.__class__):
          other = other._array

       return self.__class__(ops.add(self._array, other))


   def __mul__(self, other):

       if isinstance(other, self.__class__):
          other = other._array
 
       return self.__class__(ops.mul(self._array, other))
 





class SparseGrad(Grad): # FIXME shall we make this SparseArray instead?

   def __add__(self, other):

       array = other.copy()

       array.





"""

--- NB consider replacing tdnode.iterconv(tdnode.NodeLike)(args) and similar lines
    with smth more intuitive! (A simple if isinstance(...) could be just fine after all)


--- Make Array a NodeLike object! Just make the impl same as Point!


--- Do we still need Point? 

    When we pass non-differentiable args to fun, we'll make'em Point objs.
    In general, any differentiable args must be passed as Array objs, else
    they'll be treated as non-differentiable.
 

--- Let GraphOp.evaluate() be the only method that can retrieve Array from Node.


--- End value: remove just ONE outer layer, not all!

So maybe endval, = Args(node).concat().deshell(). 


class GraphOp: # Add new method to class GraphOp

   def evaluate(self): # Not a hidden dep after all, cuz it works like create-and-configure

       args = Args(self.end())
       args = args.deshelled()

       return args


class Args(ArgsLike): # Add new method to class Args

   def deshelled(self):

       return self.concat().deshell()


"""






"""

--- What we need to do:

* Impl backend.add_at(x, y)

* Let SparseArray.__add__ call it 

* We don't need __iadd__, it's never used... 
   
* How to handle this:

  if net_g is None: return g

  return tdwrap.functions.add(net_g, g)

* Is it OK to impl SparseArray when all we need is sparse add?
  For the rest of operations, we convert it to dense...

* The main pt of Grad obj: take care of addition!


One way: let DenseGrad impl GradLike, ArrayLike interfaces.
Let SparseGrad impl only GradLike interface.

Then, make an external function add_grads, used only for gradients!
Keep it grads.py module.


* Could also impl

GradAdd(gX, gY) -> evaluated lazily, can take other GradAdd's as args

or make it GradSum(grads), which has .add(grad) out-of-place method
use GradAdd.__call__()/evaluate() to compute the actual grad 

then we'll have

Sum:   self._grads[node] = GradAdd(grads)   

Accum: gradsum           = self._grads.get(node, GradAdd())
       self._grads[node] = gradadd.add(grad)

       or make self._grads = collections.defaultdict(GradAdd)?  

When do we evaluate GradSum? When returning it from .pick() or .result()!
Evaluate it once, cache the result.

Possibly we could make GradAdd(ArrayLike), evaluated automatically each time we access it as array!



################################## 

USE THIS!


* We need different behavior of GradAdd for SparseGrad inputs!

  Probs all the add-steps should be in [Dense/Sparse]Grad.add() method!
  We impl __radd__ method which covers other = None/0 input!
  So self._grads[node] will store [Dense/Sparse]Grad.

  We'll have DenseGrad(ArrayLike, GradLike), SparseGrad(ArrayLike, GradLike)
  --> delegate all to _array (in case of SparseGrad, create an array internally using todense() method)


* Main pt: any GradLike must impl __add__, __radd__ methods.
  Differently from Array, they will return Dense/SparseGrad object, not Array object. 


* Alternatively, we could just do DenseGrad(ArrayLike), SparseGrad(ArrayLike) since GradLike
  iface doesn't add anything new anyway! Just treat DenseGrad, SparseGrad as some impl of ArrayLike.


* Grad will be eagerly evaluated

Sum:   self._grads[node] = sum(grads)
Accum: self._grads[node] = self._grads.get(node) + grad
  

################################## 

 


* Two opts for grads:

(a) DenseGrad(ArrayLike, GradLike) -> delegate all to _array
    SparseGrad(GradLike)           -> impl add()

(b) Make all dense grads = arrays, no DenseGrad class 
    SparseGrad(ArrayLike) -> impl add(), convert to dense for all other operations

Opt (b) sounds better!


* Make sure you impl backend.add_at(array, idx, x) as an out-of-place function!  



--- Examples:

def make_sparse_object(x, idx, vs):

    if isinstance(idx, list):
       if len(idx) == 0 or not isinstance(idx[0], slice):
          idx = np.array(idx, dtype='int64')

    def add(A):
        Ac = A.copy()
        np.add.at(Ac, idx, x)
        return Ac

    return SparseObject(vs, add)


defvjp(ArrayBox.__getitem__, lambda ans, A, idx: \
                             lambda g: make_sparse_object(g, idx, vspace(A)))


class GradSum(Cumulative):

   def add(self, node, grads):

       self._grads[node] = reduce(tdmanip.add_grads, grads, None)
       self._last = node
       return self



class GradAccum(Cumulative):

   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = tdmanip.add_grads(self._grads.get(node), grad)

       return self



def add_grads(net_g, g): # TODO impl and use add() function, with @differentiable decorator 
                         #      (or overload __add__ operator to make it @differentiable)
    if net_g is None:  
       return g

    return tdwrap.functions.add(net_g, g)



def add_grads(prev_g, g):

    is_sparse = type(g) in sparse_object_types
    vs = vspace(g)

    if  prev_g is None:

        new_g = sparse_add(vs, None, g) if is_sparse else g
        return new_g

    new_g = sparse_add(vs, prev_g, g) if is_sparse else vs.add(prev_g, g)
    return new_g    


"""






































