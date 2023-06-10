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
    out     = td.zeros(inds, dtype=beta.dtype)

    for i in range(2):
      for j in range(2):
        for k in range(2):
          for l in range(2):

              if (i+j+k+l) % 2 == 0:

                 unit = td.space(out).unit(
                           (i,j,k,l)
                        )
                 elem = td.sqrt(
                           lambdas[i] * lambdas[j] * lambdas[k] * lambdas[l]
                        )
                 out = out + 0.5 * unit * elem

    return out
  



def renormalize(x, chi, eps):

    uA, sA, vA, _ = td.linalg.svd(x, linds=("u","l"), sind="s")
    uB, sB, vB, _ = td.linalg.svd(x, linds=("l","d"), sind="s")

    D   = x.shape[0]
    chi = min(
              min(D**2, chi), 
              min(
                  td.sumover(td.greater(sA, eps)).item(), 
                  td.sumover(td.greater(sB, eps)).item()
                 )
             )

    sA = sA[:chi]("s")
    sB = sB[:chi]("s")

    uA = td.transpose(uA, "u", "l", "s")[:,:,:chi]("uls")
    vA = td.transpose(vA, "d", "r", "s")[:,:,:chi]("drs")
    uB = td.transpose(uB, "l", "d", "s")[:,:,:chi]("lds")
    vB = td.transpose(vB, "u", "r", "s")[:,:,:chi]("urs")

    uA = td.contract(uA, td.sqrt(sA), product=("u","l","s"))
    vA = td.contract(vA, td.sqrt(sA), product=("d","r","s"))
    uB = td.contract(uB, td.sqrt(sB), product=("l","d","s"))
    vB = td.contract(vB, td.sqrt(sB), product=("u","r","s"))

    return td.contract(
              uB("xwu"), 
              vA("yxl"),  
              vB("yzd"), 
              uA("wzr"), 
              product="uldr"
           )

                   


def trg(beta, chi, nsteps, eps=1e-15):

    x   = ztensor(beta)
    lnZ = 0.0

    for step in range(nsteps):

        maxval = td.amax(x)
        lnZ    = lnZ + 2**(nsteps - step) * td.log(maxval)

        x = renormalize(x / maxval, chi, eps)
 
    lnZ = lnZ + td.log(td.trace(td.trace(x, ("u","d")), ("l","r")))

    return lnZ / (2**nsteps)  




def main():

    chi    = 16
    nsteps = 20

    betas = []
    lnzs  = []
    eds   = []
    cvs   = []
    
    for beta in np.linspace(0.4, 0.5, 51): 

        beta  = td.astensor(beta, dtype="float64")
        lnZ   = trg(beta, chi, nsteps)
        dlnZ  = td.gradient(trg)(beta, chi, nsteps)
        dlnZ2 = td.gradient(td.gradient(trg))(beta, chi, nsteps)

        betas.append(td.asdata(beta).item())
        lnzs.append(td.asdata(lnZ).item())
        eds.append(td.asdata(-dlnZ).item())
        cvs.append(td.asdata(dlnZ2 * beta**2).item())

        print(betas[-1], lnzs[-1], eds[-1], cvs[-1])

    plt.plot(betas, eds, marker='o', markersize=10)
    plt.savefig("trg_ising_energy_density.png")
    plt.clf()

    plt.plot(betas, cvs, marker='o', markersize=10)
    plt.savefig("trg_ising_specific_heat.png")    



#cProfile.run('main()', sort='tottime')
main()














