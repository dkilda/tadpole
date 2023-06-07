#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt

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




def ztensor(beta):

    inds = (
            IndexLit("u",2), 
            IndexLit("l",2), 
            IndexLit("d",2), 
            IndexLit("r",2),
           )

    lambdas = [2*td.cosh(beta), 2*td.sinh(beta)]
    tensor  = td.zeros(inds, dtype=beta.dtype)

    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):
              if (i+j+k+l) % 2 == 0:
                 tensor[i,j,k,l] = 0.5 * td.sqrt(
                    lambdas[i] * lambdas[j] * lambdas[k] * lambdas[l]
                 )
    return tensor
  



def renormalize(x, chi, eps):

    uA, sA, vA = td.linalg.svd(x, linds=("u","l"), sind="sa")
    uB, sB, vB = td.linalg.svd(x, linds=("l","d"), sind="sb")

    D   = x.shape[0]
    chi = min(
              min(D**2, chi), 
              min(
                  td.sumover(td.greater(sA, eps)).item(), 
                  td.sumover(td.greater(sB, eps)).item()
                 )
             )


    def _trunc(v, ind):

        v = v[td.elem({ind: slice(chi)})]
        v = td.reindex(
               v, 
               {f"{ind}_@start_{0}_@end_{chi}": IndexLit(ind, chi)}
            )

        return v


    def trunc(u, s, v, ind):

        return tuple(map(lambda x: _trunc(x, ind), (u, s, v)))


    uA, sA, vA = trunc(uA, sA, vA, "sa")
    uB, sB, vB = trunc(uB, sB, vB, "sb")

    uA = td.contract(uA, td.sqrt(sA), product=("u","l","sa"))
    vA = td.contract(vA, td.sqrt(sA), product=("d","r","sa"))

    uB = td.contract(uB, td.sqrt(sB), product=("l","d","sb"))
    vB = td.contract(vB, td.sqrt(sB), product=("u","r","sb"))

    return td.contract(
              uB("xwu"), 
              vA("yxl"),  
              vB("yzd"), 
              uA("wzr"), 
              product="uldr"
           )

                   


def trg(x, chi, nsteps, eps=1e-5):

    lnZ = 0.0

    for step in range(nsteps):

        maxval = td.amax(td.absolute(x))

        x   = x / maxval
        lnZ = lnZ + (2**(nsteps - step)) * td.log(maxval)

        x = renormalize(x, chi, eps)
 
    lnZ = lnZ + td.log(td.trace(td.trace(x, ("u","d")), ("l","r")))

    return lnZ / (2**nsteps)  




def tonumpy(xs):

    return np.concatenate([td.asdata(x).item() for x in xs])




def main():

    chi    = 16
    nsteps = 20

    betas = []
    eds   = []
    cvs   = []
    
    for beta in np.linspace(0.4, 0.5, 51):

        beta  = td.astensor(beta, dtype="float32")
        lnZ   = trg(beta, chi, nsteps)
        dlnZ  = td.gradient(trg)(beta, chi, nsteps)
        dlnZ2 = td.gradient(td.gradient(trg))(beta, chi, nsteps)
        
        betas.append(beta)
        eds.append(-dlnZ)
        cvs.append(dlnZ2 * beta**2)

    betas = tonumpy(betas)
    eds   = tonumpy(eds)
    cv    = tonumpy(cvs)

    plt.plot(betas, eds)
    plt.plot(betas, cvs)
    

    
main()



















