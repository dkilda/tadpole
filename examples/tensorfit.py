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

import scipy.optimize

import timeit
import cProfile


d = 10
D = 5
L = 4


ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for n in "udlr":
    make_index(f"D{n}", D)


for l in range(1,L+1):
    make_index(f"d{l}", d)


M    = {}
M[1] = td.randn((ind["d1"], ind["Du"], ind["Dl"]))
M[2] = td.randn((ind["d2"], ind["Du"], ind["Dr"]))
M[3] = td.randn((ind["d3"], ind["Dd"], ind["Dl"]))
M[4] = td.randn((ind["d4"], ind["Dd"], ind["Dr"]))


target = td.randuniform((ind["d1"], ind["d2"], ind["d3"], ind["d4"]))
target = target / td.amax(target)




def pack(tn):

    data = []

    for i in range(1, L+1):      

        t = td.fuse(tn[i], {tuple(td.union_inds(tn[i])): f"flat{i}"})
        
        data.append(td.asdata(t, backend="numpy").reshape(-1))

    return np.concatenate(tuple(data))




def unpack(vec, tn):

    tnc   = tn.copy()
    start = 0

    for i in range(1, L+1):   

        t = td.astensor(vec[start : start + tnc[i].size], f"flat{i}")
        t = td.split(t, {f"flat{i}": tuple(td.union_inds(tnc[i]))})

        start += tnc[i].size
        tnc[i] = td.transpose_like(t, tnc[i])

    return tnc




def distance(*ts):

    result = td.contract(*(t for t in ts))

    return td.norm(result - target, linds=(ind["d1"], ind["d2"])) 




def gradfun(x):

    tn = unpack(x, M) 
    ts = [tn[i] for i in range(1, L+1)]

    val   = distance(*ts)
    grads = [td.gradient(distance, i)(*ts) for i in range(L)]
     
    return td.asdata(val), pack(grads)





def main():

    x      = pack(M)
    result = scipy.optimize.minimize(gradfun, x, jac=True, method="L-BFGS-B")
    out    = unpack(result.x, M)

    print("minimum: ", out)
    



##########################################################################


L = 4

sx = 0.5 * np.array([[0.0,  1.0], 
                     [1.0,  0.0]])
sy = 0.5 * np.array([[0.0, -1j], 
                     [1j,   0.0]])
sz = 0.5 * np.array([[1.0,  0.0], 
                     [0.0, -1.0]])

ham = np.eye(2)
for l in range(L-1):
    ham  = np.kron(ham, np.eye(2))
    ham += np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz) 

psi = np.eigh(ham)[1][:,0]
psi = np.reshape(psi, (2,)*L)

target = td.astensor(psi, tuple(ind[f"i{l}"] for l in range(L)))




d = 2
D = 32

ind = {}
def make_index(tags, size):
    ind[tags] = IndexGen(tags, size)


for l in range(L):
    make_index(f"i{l}", d)


make_index("m0",    1)
make_index("m{L}",  1)
for l in range(L-1):
    make_index(f"m{l+1}", D)


make_index("mh0",   1)
make_index("mh{L}", 1)
for l in range(L-1):
    make_index(f"mh{l+1}", D)




mps = {}
for l in range(L):
    mps[l] = td.randn((ind[f"m{l}"], ind[f"m{l+1}"], ind[f"i{l}"]))




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

    return -td.sumover(out)**2



def gradfun(x):


    inp = td.TensorGen(x, (i,))

    mps = 
    

    out = td.evaluate_with_gradient(overlap)(mps, target)
    return td.asdata(out[0]), td.asdata(out[1])




def comp(msg, expr):

    print(msg, td.asdata(expr))




def main():

    out = scipy.optimize.minimize(gradfun, mps, jac=True, method='L-BFGS-B')
    print("optimized overlap: ", out.x)

    comp("contraction:            ", contract(M))
    comp("contraction gradient-1: ", grad(contract)(M))   
    comp("contraction gradient-2: ", grad(grad(contract))(M)) 




cpu_time = timeit.timeit(main, number=1)
print("CPUTIME: ", cpu_time)

cProfile.run('main()', sort='tottime')



