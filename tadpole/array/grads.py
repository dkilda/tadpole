#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import numpy as np

import tadpole.array.array_ops as ops
import tadpole.array.array     as array
import tadpole.array.util      as util




###############################################################################
###                                                                         ###
###  A general framework for array gradients                                ###
###                                                                         ###
###############################################################################


# --- Sparse gradient class ------------------------------------------------- #

class SparseGrad(array.ArrayLike):

   def __init__(self, space, idxs, vals):

       self._space = space
       self._idxs  = idxs
       self._vals  = vals

 
   @property
   def _array(self):

       return self._todense()


   def todense(self):

       return ops.put(self._space.zeros(), idxs, self._vals)


   def copy(self):

       return self._array.copy()


   def space(self):

       return self._space 


   def pluginto(self, funcall):

       return self._array.attach(funcall)


   def __getitem__(self, coords):

       return self._array[coords]


   @property
   def dtype(self):
       return self._space.dtype

   @property 
   def ndim(self):
       return self._space.ndim

   @property
   def shape(self):
       return self._space.shape


   def __neg__(self):

       return -self._array


   def __add__(self, other): 

       if other == 0:
          other = self._space.zeros()

       return ops.put(other, self._idxs, self._vals, accumulate=True)


   def __mul__(self, other):

       return self._array * other 


   def __radd__(self, other): 

       return self.__add__(other)  


   def __rmul__(self, other):

       return other * self._array   



"""

Move to tadpole/autodiff/grad.py:


# --- Gradient summation ---------------------------------------------------- #

class GradSum(Cumulative):

   def __init__(self, seed, grads=None):

       if grads is None:
          grads = {}

       self._last  = None
       self._seed  = seed
       self._grads = grads


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("nodes", list(self._grads.keys()))
       rep.val("grads", list(self._grads.values()))

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other)
       log.val(self._seed,  other._seed)
       log.val(self._grads, other._grads)

       return bool(log)


   def add(self, node, grads):

       self._grads[node] = sum(grads)  
       self._last = node
       return self


   def pick(self, nodes):

       if not nodes:
          return (self._seed,)   

       return tuple(map(self._grads.__getitem__, nodes))


   def result(self):

       last = self._last
       if last is None:
          last = list(self._grads)[-1]

       return self._grads[last]




# --- Gradient accumulation ------------------------------------------------- #

class GradAccum(Cumulative):

   def __init__(self, grads=None):

       if grads is None:
          grads = {}

       self._grads = grads


   def __repr__(self):

       rep = tdutil.ReprChain()

       rep.typ(self)
       rep.ref("nodes", list(self._grads.keys()))
       rep.val("grads", list(self._grads.values()))

       return str(rep)


   def __eq__(self, other):

       log = tdutil.LogicalChain()

       log.typ(self, other)
       log.val(self._grads, other._grads)

       return bool(log)


   def add(self, nodes, grads):

       for node, grad in zip(nodes, grads):
           self._grads[node] = self._grads.get(node, 0) + grad  

       return self


   def pick(self, node): 
 
       grad = self._grads.pop(node)

       self._grads[None] = grad
       return grad


   def result(self): 

       try:
          return self._grads[None] 

       except KeyError:
          last = list(self._grads)[-1]
          return self._grads[last]

"""

###############################################################################
###############################################################################
###############################################################################





       
       



"""

One more thing:

   def apply(self, fun):

       try:
            last = self.packs().last()
       except StopIteration:
            last = self.packs().first()

       args = last.deshell() 
       out  = fun(*(arg.tovalue() for arg in args))

       return out   


In out = fun(*(arg.tovalue() for arg in args)),
what if some args are points, and others are arrays?

we need to make points accessible for evaluation!

e.g. suppose you want to call reshape(x, shape):
shape will be converted to Point. How do we access it?

Args convert all inputs to NodeLike's.
Probs it could convert'em back too?


* Q: Why convert all vals to Points?

* A: Cuz we need polymorphic behavior in .concat() and then inside Concatenation.


##############

USE THIS!


--- let the last deshelling to remove the final shell from all args and convert them to values?


def deshell(self):

    if self.layer() == minlayer():
       return Args(tuple(self._sources)) --> but Args will convert it back to Points...

    ... 


Let Args:

   self._args = args

   .nodelike(self): convert all args to nodelikes, use this in .concat()

   this way Args will keep the original vals!


--- Ok, how does this change the deshelling iteration in Envelope?

    A: we want to deshell when we hit minlayer()

    But we do that anyway! We call last.deshell() in .apply()! So no changes.
    This last deshell will return args with no-Point values.


--- One last thing: is the usage of __iter__ of Args dependent on the NodeLike-ness of args?



##############



But: Concatenation knows which args are differentiable by looking at their layers.
     It will skip any Points anyway.


"""





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






































