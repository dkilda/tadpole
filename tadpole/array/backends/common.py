#!/usr/bin/env python3
# -*- coding: utf-8 -*-




###############################################################################
###                                                                         ###
###  Creation methods                                                       ###
###                                                                         ###
###############################################################################


# --- Create unit array ----------------------------------------------------- #

def unit(self, shape, idx, **opts):

    array      = self.zeros(shape, **opts)
    array[idx] = 1

    return array




###############################################################################
###                                                                         ###
###  Eigenvalue solvers                                                     ###
###                                                                         ###
###############################################################################


# --- General eigenvalue solver --------------------------------------------- #

def eig(self, fun, x):

    if  x.shape[0] > x.shape[1]:

        # Do eigendecomposition of xH*x = V*S2*VH, construct U, S, VH 
        # (use 'clip' to set small negative evals in S2 to 0)
        S2, V = fun(self.dot(self.htranspose(x, axes=(1,0)), x))

        U  = self.dot(x, V)
        VH = self.htranspose(V, axes=(1,0))
        S2 = self.clip(S2, 0, None)
           
        S = S2**0.5
        U = U / self.reshape(S, (1, -1)) 

    else: 

        # Do eigendecomposition of x*xH = U*S2*UH, construct U, S, VH 
        # (use 'clip' to set small negative evals in S2 to 0)
        S2, U = fun(self.dot(x, self.htranspose(x, axes=(1,0))))

        VH = self.dot(self.htranspose(U, axes=(1,0)), x)  
        S2 = self.clip(S2, 0, None)
           
        S  = S2**0.5
        VH = VH / self.reshape(S, (-1, 1)) 


    # Sort svals/svecs in descending order 
    # (flip axes 1, 0, 0 of U, S, VH arrays respectively) 
    U  = self.flip(U,  (1,))
    S  = self.flip(S,  (0,))
    VH = self.flip(VH, (0,))

    return U, S, VH




# --- Hermitian matrix eigenvalue solver ------------------------------------ #

def eigh(self, fun, x):

    # Eigendecomposition of Hermitian matrix x = U*S*UH
    S, U = fun(x)

    # Sort evals in descending order
    S, U = S[::-1], U[:, ::-1]

    # Since x = xH, define VH <- UH * sgn(S), S <- S * sgn(S) = abs(S) 
    VH = self.reshape(self.sign(S), (-1, 1)) * self.htranspose(U, axes=(1,0))
    S  = self.abs(S)

    return U, S, VH

    


















































