# /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import functools
import numpy as np
from sklearn import covariance
from abc import ABCMeta, abstractmethod
import copy
from pymtl.misc import numerics
import pdb
import pymanopt.manifolds as Manifolds
from pymanopt import Problem
import pymanopt.solvers as solvers
import autograd.numpy as autonp

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class PriorParamsInterface(object):
    """
    Abstract class interface to update prior parameters as used by MTL algorithms
    """
    __metaclass__ = ABCMeta


    @abstractmethod
    def update_params(self, samples):
        """
        Updates the parameters of this prior based on the given samples
        """
        pass

    @abstractmethod
    def diff(self, other):
        """
        Computes some sort of difference measure of this prior from another one. If the priors are
        similar to each other, values near zero should be returned.
        """
        pass


class GaussianParams(PriorParamsInterface):
    """
    Class implementing standard maximum likelihood updates for vector-variate Gaussian distributions
    """

    def __init__(self, dim, norm_style='ML', init_mean_val=0, init_var_val=1):
        """
        dim: Dimension of input (scalar)
        norm_style: 'ML' -- standard covariance estimation
                    'Trace' -- standard covariance estimation, scaled by the trace
        """
        self.mu = init_mean_val * np.ones((dim, 1))
        self.Sigma = init_var_val * np.eye(dim)
        self._id = np.eye(dim)
        self._norm_style = norm_style

    def update_params(self, samples):
        """
        TODO
        """
        self.mu = self.estimate_mean(samples)
        self.Sigma = self.estimate_cov(samples, self.mu)

    def diff(self, other):
        """
        TODO
        """
        if not isinstance(other, GaussianParams):
            raise ValueError('Given instance is not of type {}'.format(type(self)))
        d1 = np.abs(self.mu - other.mu)
        d2 = np.abs(self.Sigma - other.Sigma)
        return np.sum(d1) + np.sum(d2)

    def estimate_mean(self, samples):
        """
        Estimate the mean of the samples
        """
        d = samples[0].shape[0]
        mu = np.zeros((d, 1))
        for t in range(len(samples)):
            mu = mu + samples[t]
        return (1.0/len(samples))*mu

    def estimate_cov(self, samples, mean):
        """
        Estimate the empirical covariance of the weight vectors, possibly
        with regularization. 
        """
        d = mean.shape[0]
        # Accumulate statistics
        Sigma = np.zeros((d, d))
        for t in range(len(samples)):
            zm = samples[t] - mean
            Sigma = Sigma + zm.dot(zm.T)
        # Normalize factor of estimate
        if self._norm_style == 'ML':
            norm = 1.0/(len(samples)-1)
        elif self._norm_style == 'Trace':
            norm = 1.0/np.trace(Sigma)
        else:
            raise ValueError('Norm style {} not known'.format(self._norm_style))
        Sigma = norm*Sigma
        # Add diagonal loading term
        self.diag_eps = 0.1*np.mean(np.abs(np.linalg.eig(Sigma)[0])) # TODO
        return Sigma + self.diag_eps*self._id

class SKGaussianParams(GaussianParams):
    """
    TODO
    """

    def __init__(self, dim, estimator='OAS', **kwargs):
        """
        TODO
        """
        super(SKGaussianParams, self).__init__(dim, **kwargs)
        if estimator == 'EmpiricalCovariance':
            self._estimator = covariance.EmpiricalCovariance(assume_centered=True)
        elif estimator == 'LedoitWolf':
            self._estimator = covariance.LedoitWolf(assume_centered=True)
        elif estimator == 'MinCovDet':
            self._estimator = covariance.MinCovDet(assume_centered=True)
        elif estimator == 'OAS':
            self._estimator = covariance.OAS(assume_centered=True)
        elif estimator == 'ShrunkCovariance':
            self._estimator = covariance.ShrunkCovariance(assume_centered=True)
        else:
            raise ValueError('Unknown estimator: {}'.format(estimator))

    def estimate_cov(self, samples, mean):
        """
        TODO
        """
        samples = np.squeeze(np.array(samples)) - mean.T
        self._estimator.fit(samples)
        #self.diag_eps = 0.1*np.mean(np.abs(np.linalg.eig(self._estimator.covariance_)[0])) # TODO
        return self._estimator.covariance_ # + self.diag_eps*self._id


class LowRankGaussianParams(SKGaussianParams):
    """
    Interface that implements a trace penalty on the prior mean, which results in a different mean update method.
    """

    def __init__(self, dim, nu=1, conv_tol=1e-4, max_its=50, k=5, **kwargs):
        """
        TODO
        """
        super(LowRankGaussianParams,self).__init__(dim, **kwargs)
        self.nu = nu
        self.conv_tol = conv_tol
        self.max_its = max_its
        n = int((-1.0+np.sqrt(1+8*dim))/2)
        self.manifold = Manifolds.PSDFixedRank(n,k)
        self.E = numerics.generate_elimination_matrix(n)
        self.D = numerics.generate_duplication_matrix(n)
        self.Tnn = numerics.generate_vectranspose_matrix(n,n)
        self.lastpt = self.manifold.rand()

    def update_params(self, samples):
        """
        Given samples, iteratively update mu and sigma until convergence
        """
        M = [numerics.unvech(s,self.D) for s in samples]
        print('Number of samples that are PSD: {}'.format(
            functools.reduce(lambda x,y: x + int((np.linalg.eig(y)[0] >= 0).all()), M, 0)))
        diff = 1e10
        it = 0
        while diff > self.conv_tol and it < self.max_its:
            prev_prior = copy.deepcopy(self)
            #pdb.set_trace()
            self.mu = self.estimate_mean(samples, self.Sigma)
            self.Sigma = self.estimate_cov(samples, self.mu)
            print('sanity: covariance is PD: {}'.format((np.linalg.eig(self.Sigma)[0] > 0).all()))
            diff = self.diff(prev_prior)
            it += 1
            print('Prior iteration {}, difference {}'.format(it, diff))
        U = numerics.unvech(self.mu, self.D)
        print('Rank of prior matrix: {}/{}'.format(np.linalg.matrix_rank(U),U.shape[0]))

    def diff(self, other):
        """
        TODO
        """
        if not isinstance(other, LowRankGaussianParams):
            raise ValueError('Given instance is not of type {}'.format(type(self)))
        #return self.manifold.dist(self.lastpt, other.lastpt)
        return super(LowRankGaussianParams,self).diff(other)

    def estimate_mean_explicit(self, samples, Sigma):
        """
        Estimate mean given trace norm penalty (and lemma from Farquhar) 
        [not sure if makes sense and not working...]
        """
        Sinv = np.linalg.inv(Sigma)
        E = self.E
        nu = self.nu
        mu_hat = np.sum(np.asarray(samples).squeeze(),axis=0)
        n = len(samples)
        Tnn = self.Tnn
        def cost(L):
            vecL = np.reshape(L,(-1,1))
            vecLLT = np.reshape(np.dot(L,L.T),(-1,1))
            costsum = 0
            for x in samples:
                dx = x - E.dot(vecLLT)
                costsum += np.dot(dx.T, np.dot(Sinv,dx))
            costsum *= 0.5
            costsum += nu*np.dot(vecL.T,vecL)
            return costsum
            
        def d_cost(L):
            vecL = np.reshape(L,(-1,1))
            vecLLT = np.reshape(np.dot(L,L.T),(-1,1))
            T = -Sinv.dot(mu_hat.reshape((-1,1))) + n*Sinv.dot(E.dot(vecLLT))
            D_vecLLT = ( Tnn + np.eye(len(vecLLT))).dot(np.kron(L,np.eye(L.shape[0])))
            return (T.T.dot(E.dot(D_vecLLT)) + nu*vecL.T).reshape(L.shape)

        solver = solvers.ConjugateGradient()
        prob = Problem(manifold=self.manifold,cost=cost, egrad=d_cost)
        
        Lopt = solver.solve(prob)
        
        return E.dot(np.reshape(Lopt.dot(Lopt.T),(-1,1)))

    def estimate_mean(self, samples, Sigma):
        '''
        Function that just does autodiff and a straight trace constraint"
        '''
        E = self.E
        nu = self.nu
        Sinv = np.linalg.inv(Sigma)
        def cost(M):
            W = autonp.dot(M,autonp.transpose(M))
            vecW = autonp.reshape(W,(-1,1))
            diff = [s - autonp.dot(E,vecW) for s in samples]
            c  = 0
            for d in diff:
                c += 0.5 * autonp.dot(autonp.transpose(d),autonp.dot(Sinv,d))
            c += autonp.linalg.norm(M, ord='fro')
            return c
        
        solver = solvers.SteepestDescent()
        prob = Problem(manifold=self.manifold,cost=cost,verbosity=1)
        
        V = solver.solve(prob, x=self.lastpt)
        self.lastpt = V
        Wopt = V.dot(V.T)
        assert((Wopt == Wopt.T).all()) #force the result to be symmetric
        return numerics.vech(Wopt,self.E).reshape((-1,1))

class TemporalGP(SKGaussianParams):
    '''
    Interface that maps inputs to FIR filters and reprojects before parameter updates
    '''

    def __init__(self, dim, **kwargs):
        super(TemporalGP,self).__init__(dim,**kwargs)

    def update_params(self, samples):
        b = [numerics.solve_fir_coef(s) for s in samples]
        super(TemporalGP,self).update_params(
            [np.asarray([f(*c) for f in flist]).reshape((c.shape[0],1)) for c, flist in b]
        )
        
        self.meanfilter = numerics.solve_fir_coef(self.mu)[0]
