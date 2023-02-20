#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc



class ArrayLike(abc.ABC):
   pass



class Type(abc.ABC):
   pass




class ArrayType(Type):

   def __add__(self, other):

       if not other:
          return self

       if self == other:
          return self     

       return ValueError((f"ArrayType.__add__: "
                          f"cannot combine unequal array types")) 


   def __radd__(self, other):

       return self.__add__(other)


   def __call__(self, *args, **kwargs):

       return Array(*args, **kwargs)






class ArrayFun:

   def __init__(self, backend, shape, fun):

       self._backend = backend
       self._shape   = shape
       self._fun     = fun


   def __call__(self, *args, **kwargs):

       pass 












# --- Array ----------------------------------------------------------------- #

class Array(ArrayLike):

   def __init__(self, data, backend=None):

       self._data    = data
       self._backend = backend


   def type(self):

       return ArrayType()


   def copy(self):

       return self.__class__(self._data, self._backend)


   def with_backend(self, backend):

       return self.__class__(self._data, backend)


   def pluginto(self, engine):

       return engine.attach(self, self._data, self._backend)


   def __getitem__(self, coords):

       return self._array[coords]


   @property
   def dtype(self):
       return self._backend.dtype(self._array)

   @property 
   def ndim(self):
       return self._backend.ndim(self._array)

   @property
   def shape(self):
       return self._backend.shape(self._array)




class Engine(abc.ABC):

   @abc.abstractmethod
   def attach(self, array, data, backend):
       pass







class UnaryEngine(Engine):

   def __init__(self, array=None, data=None, backend=None):

       self._array   = array
       self._data    = data
       self._backend = backend


   def attach(self, array, data, backend):

       return self.__class__(array, data, backend)


   def apply(self, fun):

       data = fun(self._backend, self._data)

       return self._array.type(data, self._backend)




class Shape(Engine):

   def __init__(self, engine): # FIXME consider using some NullArray instead

       self._engine = engine


   def attach(self, array, data, backend): # FIXME extract this into UnaryEngine class, include it by composition

       return self.__class__(self._engine.attach(array, data, backend))


   def reshape(self, shape):

       return self._engine.apply(lambda b,x: b.reshape(x, shape))




def reshape(x, shape):

    engine = x.pluginto(Shape())
    return engine.reshape(shape)


def reshape(x, shape):

    engine = x.pluginto(Reshape())
    return engine.compute(shape)




class UnaryMath(Engine):

   def __init__(self, engine=None):

       if engine is None:
          engine = UnaryEngine()

       self._engine = engine


   def attach(self, array, data, backend):

       return self.__class__(self._engine.attach(array, data, backend))  


   def negative(self):

       return self._engine.apply(lambda b,x: b.negative(x))


   def sin(self):

       return self._engine.apply(lambda b,x: b.sin(x))


   def cos(self):

       return self._engine.apply(lambda b,x: b.cos(x))





class BinaryEngine(Engine):

   def __init__(self, content):

       if content is None:
          content = tdutil.Sequence() # TODO replace with Pair()

       self._content = content


   def attach(self, array, data, backend):

       return self.__class__(self._content.push((array, data, backend)))


   def apply(self, fun):

       arrays, datas, backends = zip(*self._content)

       arraytype = sum(array.type() for array in arrays)
       backend   = backends[0]

       return arraytype(fun(backend, *datas), backend)






class BinaryMath(Engine):   

   def __init__(self, engine=None):

       if engine is None:
          engine = BinaryEngine()

       self._engine = engine


   def attach(self, array, data, backend):

       return self.__class__(self._engine.attach(array, data, backend))


   def add(self):

       return self._engine.apply(lambda b,x,y: b.add(x,y))
 

   def mul(self):

       return self._engine.apply(lambda b,x,y: b.mul(x,y))


   def dot(self):

       return self._engine.apply(lambda b,x,y: b.dot(x,y))




def mul(x, y):

    engine = BinaryMath()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)

    return engine.mul()


def mul(x, y):

    engine = Mul()
    engine = x.pluginto(engine)
    engine = y.pluginto(engine)

    return mul.compute()




def mul(x, y):

    engine = Mul(Binary(x, y))
    return engine.apply()
 



class Mul: # FIXME just move everyth to mul() function?

   def __init__(self, engine):

       self._engine = engine


   def apply(self):

       return self._engine.apply(lambda b,x,y: b.mul(x,y))



class BinaryCall:

   def __init__(self, x, y):

       self._x = x 
       self._y = y


   def apply(self, fun): 

       engine = NaryEngine(fun)

       engine = self._x.pluginto(engine)
       engine = self._y.pluginto(engine)

       return engine





   def apply(self, fun):

       arrays, datas, backends = zip(*self._content)

       arraytype = sum(array.type() for array in arrays)
       backend   = backends[0]

       return arraytype(fun(backend, *datas), backend)






class NaryEngine(Engine):

   def __init__(self, content):

       if content is None:
          content = tdutil.Sequence() 

       self._content = content


   def attach(self, array, data, backend):

       return self.__class__(self._content.push((array, data, backend))


   def apply(self, fun):

       arrays, datas, backends = zip(*self._content)

       arraytype = sum(array.type() for array in arrays)
       backend   = backends[0]

       return arraytype(fun(backend, *datas), backend)




class Einsum(Engine):

   def __init__(self, engine=None):

       if engine is None:
          engine = NaryEngine()

       self._engine = engine


   def attach(self, array, data, backend):

       return self.__class__(self._engine.attach(array, data, backend))


   def compute(self, equation, optimize=True):

       def fun(b, *xs):
           return b.einsum(equation, *xs, optimize=optimize)

       return self._engine.apply(fun)


         




































