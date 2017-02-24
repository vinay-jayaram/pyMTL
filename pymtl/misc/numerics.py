# /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.covariance import oas
from pymtl.misc import verbose as vb
import sympy 
import scipy.optimize as optimize

#from fancyimpute import SoftImpute, NuclearNormMinimization

__author__ = "Karl-Heinz Fiebig, Vinay Jayaram"
__copyright__ = "Copyright 2017"


def complete_matrix(X, lam=1.0, beta=1, sigma=1, Z=None, max_iter=200, tol=1e-9, 
                    initial_rank=None, verbose=False):
    if initial_rank is None:
        initial_rank = max(1, X.shape[0]*0.2)
    if verbose:
        vb.pyout('Original rank: {}'.format(np.linalg.matrix_rank(X)))
    if Z is None:
        Z = low_rank_approx(X, r=int(initial_rank))
    if verbose:
        vb.pyout('Initial rank of approximation: {}'.format(np.linalg.matrix_rank(Z)))
    U_X, d_X, V_X = np.linalg.svd(X)
    # Setup parameters
    lam = lam * np.max(d_X) * 0.999999999999
    a = lam * beta
    b = beta
    # Loss function
    def loss(Z):
        _, s, _ = np.linalg.svd(Z)
        err = 1.0/(2*sigma)*np.linalg.norm(Z, ord='fro')
        reg = np.sum((a+1)*np.log(b+s))
        return err + reg
    loss_old = loss(Z)
    if verbose:
        vb.pyout('Initial loss: {}'.format(loss_old))
    for iter in range(max_iter):
        # E-step
        _, d_Z, _ = np.linalg.svd(Z)
        omega = (a+1) / (b+d_Z)
        # M-step
        #D_omega = np.diag(d_X - lam)
        D_omega = np.diag(d_X - omega)
        D_omega[D_omega < 0] = 0
        Z = U_X.dot(D_omega.dot(V_X))
        loss_current = loss(Z)
        conv = np.abs(loss_old - loss_current) / loss_old
        if verbose:
            vb.pyout('Iteration {}: Current loss {} / Convergence {}'.format(iter, loss_current, conv))
        if loss_current == 0 or conv < tol:
            break
        loss_old = loss_current
    if verbose:
        vb.pyout('Final Z rank: {}'.format(np.linalg.matrix_rank(Z)))
    return Z


def low_rank_approx(X, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD

    Taken from https://gist.github.com/thearn/5424219
    """
    u, s, v = np.linalg.svd(X, full_matrices=False)
    Z = np.zeros((len(u), len(v)))
    for i in range(r):
        Z += s[i] * np.outer(u.T[i], v[i])
    return Z


def vec(X, stack_cols=True):
    if stack_cols:
        return X.T.flatten()
    else:
        return X.flatten()


def unvec(v, cols_stacked=True):
    v = v.flatten()
    d = int(np.sqrt(len(v)))
    if cols_stacked:
        return v.reshape(d, d).T
    else:
        return v.reshape(d, d)


def vech_kh(X, stack_cols=True, conserve_norm=False):
    assert X.shape[0] == X.shape[1]
    # Scale off-diagonal indexes if norm has to be preserved
    d = X.shape[0]
    if conserve_norm:
        # Scale off-diagonal
        tmp = np.copy(X)
        triu_scale_idx = np.triu_indices(d, 1)
        tmp[triu_scale_idx] = np.sqrt(2) * tmp[triu_scale_idx]
    else:
        tmp = X
    triu_idx_r = []
    triu_idx_c = []
    # Find appropriate indexes
    if stack_cols:
        for c in range(0, d):
            for r in range(0, c+1):
                triu_idx_r.append(r)
                triu_idx_c.append(c)
    else:
        for r in range(0, d):
            for c in range(r, d):
                triu_idx_r.append(r)
                triu_idx_c.append(c)
    # Extract and return upper triangular
    triu_idx = (triu_idx_r, triu_idx_c)
    return tmp[triu_idx]


def unvech_kh(v, cols_stacked=True, norm_conserved=False):
    # Restore matrix dimension and add triangular
    v = v.flatten()
    d = int(0.5 * (np.sqrt(8 * len(v) + 1) - 1))
    X = np.empty((d, d))
    triu_idx_r = []
    triu_idx_c = []
    # Find appropriate indexes
    if cols_stacked:
        for c in range(0, d):
            for r in range(0, c+1):
                triu_idx_r.append(r)
                triu_idx_c.append(c)
    else:
        for r in range(0, d):
            for c in range(r, d):
                triu_idx_r.append(r)
                triu_idx_c.append(c)
    # Restore original matrix
    triu_idx = (triu_idx_r, triu_idx_c)
    X[triu_idx] = v
    X[np.tril_indices(d)] = X.T[np.tril_indices(d)]
    # Undo rescaling on off diagonal elements
    if norm_conserved:
        X[np.triu_indices(d, 1)] /= np.sqrt(2)
        X[np.tril_indices(d, -1)] /= np.sqrt(2)
    return X

def generate_elimination_matrix(n):
    n_vec = n**2
    n_vech = int(n*(n+1)/2)
    E = np.zeros((n_vech,n_vec))
    n_curr = n
    row = 0
    col = 0
    while n_curr > 0:
        # add n keep columns
        E[row:(row+n_curr),col:(col+n_curr)] = np.eye(n_curr);
        row = row+n_curr; col = col+n_curr;

        # add n-n_curr+1 zero columns
        n_zero = n - n_curr+1;
        col = col+n_zero;

        # reduce n_curr
        n_curr = n_curr - 1;
    return E

def generate_duplication_matrix(n):
    n_vec = n**2;
    n_vech = int(n*(n+1)/2)
    D = np.zeros((n_vech,n_vec));
    n_curr = n
    row = 0
    col = 0
    while n_curr > 0:
        # add n keep columns
        D[row:(row+n_curr),col:(col+n_curr)] = np.eye(n_curr);

        # for the 2-end dimensions ad the rest
        for k in range(1,n_curr):
            D[row+k,col+k*n] = 1;

        row = row+n_curr
        col = col+n_curr

        # add n-n_curr+1 zero columns
        n_zero = n - n_curr+1
        col = col+n_zero;

        # reduce n_curr
        n_curr = n_curr - 1;

    return D.T

def generate_vectranspose_matrix(n, k):
    '''
    Don't want to figure out closed form so I'm doing it the dumb way.
    Matrix that gives a permutation matrix that will give the vectorized transpose
    of a given vectorized matrix. 
    '''

    # generate matrix of appropriate size with numbered entries 
    indices = np.arange(n*k).reshape((n,k))
    ind_vec = np.reshape(indices,(-1,1))
    ind_T_vec = np.reshape(indices.T,(-1,1))
    Tnk = np.zeros((n*k,n*k))
    for i in range(n*k):
        Tnk[i,np.where(ind_vec == ind_T_vec[i])] = 1
    return Tnk

def vech(X, E=None):
    if E is None:
        E = generate_elimination_matrix(X.shape[0])
    vecX = X.flatten()
    return E.dot(vecX).flatten()

def unvech(x, D=None):
    n = n = int((-1.0+np.sqrt(1+8*x.shape[0]))/2)
    if D is None:
        D = generate_duplication_matrix(n)
    return D.dot(x).reshape((n,n))

def solve_fir_coef(w):
    '''
    Solve for FIR coefficients given regression coefficients for autocovariance matrices
    '''
    ndim = len(w)
    # strategy: start at the back and substitute up, then recursively calculate coefficients
    b = sympy.symarray('b',ndim)
    matb = sympy.Matrix(b)
    eqns = [b[i:].T.dot(b[:(ndim-i)]) - w[i] for i in range(ndim)]
    # generate vector function and jacobian
    f = sympy.Matrix(eqns)
    J_f = f.jacobian(matb)
    # lambdify
    F = sympy.lambdify(b, f)
    J_F = sympy.lambdify(b ,J_f)

    sol = optimize.root(lambda x: F(*x).ravel(),
                        np.random.rand(len(w)),
                        jac=lambda x: J_F(*x),
                        method='lm')
    #print(sol)
    return sol.x/sol.x[0],[sympy.lambdify(b,b[i:].T.dot(b[:(ndim-i)])) for i in range(ndim)]
