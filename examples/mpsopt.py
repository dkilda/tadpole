#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy   as np
import tadpole as td

from tadpole import (
   IndexGen,
   IndexLit,
)

from util import (
   tprint,
   TensorCollection,
   ArgsTensorCollection,
   ArgsTensor,
   Optimize,
)

import timeit
import cProfile


L = 5

sx = 0.5 * np.array([[0.0,  1.0], 
                     [1.0,  0.0]], dtype='complex128')
sy = 0.5 * np.array([[0.0, -1j], 
                     [1j,   0.0]], dtype='complex128')
sz = 0.5 * np.array([[1.0,  0.0], 
                     [0.0, -1.0]], dtype='complex128')
eye = np.eye(2, 
             dtype='complex128')


def kron(x, y, ix, iy):

    def site(i):
        if i == ix: return x
        if i == iy: return y
        return eye

    out = site(0)
    for i in range(1,L):
        out = np.kron(out, site(i))

    return out


ham = 0
pos = list(range(L))

for i,j in zip(pos, pos[1:] + pos[:1]):

    ham += kron(sx, sx, i, j)
    ham += kron(sy, sy, i, j)
    ham += kron(sz, sz, i, j) 

ham = np.real(ham)


psi = np.linalg.eigh(ham)[1][:,0]
psi = np.reshape(psi, (2,)*L)



d = 2
D = 16

ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for l in range(L):
    make_index(f"i{l}", d)


make_index("m0",    1)
make_index(f"m{L}",  1)
for l in range(L-1):
    make_index(f"m{l+1}", D)


make_index("mh0",   1)
make_index(f"mh{L}", 1)
for l in range(L-1):
    make_index(f"mh{l+1}", D)


mps = {}
for l in range(L):
    mps[l] = td.randn((ind[f"m{l}"], ind[f"m{l+1}"], ind[f"i{l}"]))

mps = TensorCollection(mps)

target = td.astensor(psi, tuple(ind[f"i{l}"] for l in range(L)))




def hc(mps):

    out = {}
    for l in range(L):
        indmap = {
                  ind[f"m{l}"]:   ind[f"mh{l}"], 
                  ind[f"m{l+1}"]: ind[f"mh{l+1}"],
                 }
        out[l] = td.reindex(mps[l].C, indmap)

    return out




def normalize(mps):

    c   = td.sqrt(norm(mps))**(1./L)
    out = {}
    for l in range(L):
        out[l] = mps[l] / c

    return out




def norm(mps):

    mpsH = hc(mps)

    out = td.ones((ind[f"m0"], ind[f"mh0"]))
    for l in range(L):
        out = (out @ mps[l]) @ mpsH[l]


    return td.sumover(out)




def overlap(mps, target):

    mps  = normalize(mps)
    mpsH = hc(mps)

    out = target
    for l in range(L):
        out = mpsH[l] @ out

    return td.absolute(td.sumover(out))




def overlapfun(*ts):

    mps = dict(zip(range(len(ts)), ts))

    return -overlap(mps, target)**2




def main():

    tprint("initial overlap: ", overlap(mps, target))

    optimize = Optimize(overlapfun, ArgsTensorCollection(mps))
    mpsout   = optimize(
                  method='CG', 
                  options={"maxiter": 100}
               ).astensorcollection()

    tprint("optimized overlap: ", overlap(mpsout, target))




cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

#cProfile.run('main()', sort='cumtime')



