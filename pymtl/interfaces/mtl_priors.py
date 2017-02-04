# /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from sklearn import covariance
from abc import ABCMeta, abstractmethod

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
    TODO
    """

    def __init__(self, dim, norm_style='ML', init_mean_val=0, init_var_val=1):
        """
        TODO
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
        TODO
        """
        d = samples[0].shape[0]
        mu = np.zeros((d, 1))
        for t in range(len(samples)):
            mu = mu + samples[t]
        return (1.0/len(samples))*mu

    def estimate_cov(self, samples, mean):
        """
        TODO
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

class SKGaussianParams(PriorParamsInterface):
    """
    TODO
    """

    def __init__(self, dim, estimator='EmpiricalCovariance', init_mean_val=0, init_var_val=1):
        """
        TODO
        """
        self.mu = init_mean_val * np.ones((dim, 1))
        self.Sigma = init_var_val * np.eye(dim)
        self._id = np.eye(dim)
        if estimator == 'EmpiricalCovariance':
            self._estimator = covariance.EmpiricalCovariance()
        elif estimator == 'LedoitWolf':
            self._estimator = covariance.LedoitWolf()
        elif estimator == 'MinCovDet':
            self._estimator = covariance.MinCovDet()
        elif estimator == 'OAS':
            self._estimator = covariance.OAS()
        elif estimator == 'ShrunkCovariance':
            self._estimator = covariance.ShrunkCovariance()
        else:
            raise ValueError('Unknown estimator: {}'.format(estimator))

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
        if not isinstance(other, SKGaussianParams):
            raise ValueError('Given instance is not of type {}'.format(type(self)))
        d1 = np.abs(self.mu - other.mu)
        d2 = np.abs(self.Sigma - other.Sigma)
        return np.sum(d1) + np.sum(d2)

    def estimate_mean(self, samples):
        """
        TODO
        """
        d = samples[0].shape[0]
        mu = np.zeros((d, 1))
        for t in range(len(samples)):
            mu = mu + samples[t]
        return (1.0/len(samples))*mu

    def estimate_cov(self, samples, mean):
        """
        TODO
        """
        self._estimator.fit(np.squeeze(np.array(samples)))
        #self.diag_eps = 0.1*np.mean(np.abs(np.linalg.eig(self._estimator.covariance_)[0])) # TODO
        return self._estimator.covariance_ # + self.diag_eps*self._id


class MatrixGaussianParams(PriorParamsInterface):
    """
    TODO
    """

    def __init__(self, dim, init_mean_val=0, init_var_val=1):
        """
        TODO
        """
        self.Mu = init_mean_val * np.ones(dim)
        self.Sigma_r = init_var_val * np.eye(dim[0])
        self.Sigma_c = init_var_val * np.eye(dim[1])
        self._id_r = np.eye(dim[0])
        self._id_c = np.eye(dim[1])

    def update_params(self, samples):
        """
        TODO
        (estimate according to: AN EXPECTATION-MAXIMIZATION ALGORITHM FOR THE MATRIX
        NORMAL DISTRIBUTION,  HUNTER GLANZ AND LUIS CARVALHO)
        """
        max_iter = 1000
        tol = 1e-4
        self.Mu = self.estimate_mean(samples)
        for iter in range(max_iter):
            Sigma_r_prev = self.Sigma_r
            Sigma_c_prev = self.Sigma_c
            self.Sigma_r = self.estimate_cov_r(samples, self.Mu, self.Sigma_c)
            self.Sigma_r = 1.0/(self.Sigma_r[0, 0]) * self.Sigma_r
            self.Sigma_c = self.estimate_cov_c(samples, self.Mu, self.Sigma_r)
            self.Sigma_c = 1.0/(self.Sigma_c[0, 0]) * self.Sigma_c
            # Normalize estimates by top left entry
            #self.Sigma_c = 1.0/(self.Sigma_c[0, 0]) * self.Sigma_c
            #self.Sigma_r = 1.0/(self.Sigma_r[0, 0]) * self.Sigma_r
            # Estimate overall covariance scale
            #beta = self._estimate_cov_rcale(samples, self.Mu, self.Sigma_r, self.Sigma_c)
            # Normalize covariances by overall scale
            #self.Sigma_c = sigma2 * self.Sigma_c
            #self.Sigma_r = sigma2 * self.Sigma_r
            # Check for convergence
            if np.allclose(self.Sigma_r, Sigma_r_prev, atol=tol) and np.allclose(self.Sigma_c, Sigma_c_prev, atol=tol):
                break
        #beta = self._estimate_cov_rcale(samples, self.Mu, self.Sigma_r, self.Sigma_c)
        #self.Sigma_r = beta*self.Sigma_r
        #print beta
        #print self.Sigma_c
        #exit(0)

    def diff(self, other):
        """
        TODO
        """
        if not isinstance(other, MatrixGaussianParams):
            raise ValueError('Given instance is not of type {}'.format(type(self)))
        d1 = np.abs(self.Mu - other.Mu)
        d2 = np.abs(self.Sigma_r - other.Sigma_r)
        d3 = np.abs(self.Sigma_c - other.Sigma_c)
        return np.sum(d1) + np.sum(d2) + np.sum(d3)

    def estimate_mean(self, samples):
        """
        TODO
        """
        mu = np.zeros(samples[0].shape)
        for samp in samples:
            mu = mu + samp
        return (1.0/len(samples))*mu

    def estimate_cov_r(self, samples, mean, cov_c):
        """
        TODO
        each samples: p x q
        cov_r: p x p
        cov_c: q x q
        """
        invCov_c = np.linalg.inv(cov_c)
        p = samples[0].shape[0]
        q = samples[0].shape[1]
        cov_r = np.zeros((p, p))
        for samp in samples:
            zm = samp - mean
            cov_r = cov_r + zm.dot(invCov_c.dot(zm.T))
        #diag_eps = max(1e-4, np.min(np.abs(np.linalg.eig(cov_r)[0])))
        return 1.0/(q*len(samples)) * cov_r #+ diag_eps*self._id_r

    def estimate_cov_c(self, samples, mean, cov_r):
        """
        TODO
         each samples: p x q
         cov_r: p x p
         cov_c: q x q
         """
        invcov_r = np.linalg.inv(cov_r)
        p = samples[0].shape[0]
        q = samples[0].shape[1]
        cov_c = np.zeros((q, q))
        # Compute covariance
        for samp in samples:
            zm = samp - mean
            cov_c = cov_c + zm.T.dot(invcov_r.dot(zm))
        # Scale estimate
        #diag_eps = max(1e-4, np.min(np.abs(np.linalg.eig(cov_c)[0])))
        return 1.0/(p*len(samples)) * cov_c# + diag_eps*self._id_c

    def _estimate_cov_rcale(self, samples, mean, cov_r, cov_c):
        invKronSigma = np.linalg.inv(np.kron(cov_c, cov_r)) # or swith kronecker operands?
        p = samples[0].shape[0]
        q = samples[0].shape[1]
        beta = 0
        for samp in samples:
            zm = (samp - mean).T.flatten(order='C') # Vectorized zero mean data
            beta = beta + zm.T.dot(invKronSigma.dot(zm))
        beta = 1.0/(p*q*len(samples)) * beta
        return beta

class MatrixGaussianKronParams(PriorParamsInterface):
    """
    Multivariate equivalent to the Matrix Gaussian. In contrast to MatrixGaussianParams, this
    version does not have to perform iterative updates between the row- and column covariance.
    Instead, the kronecker product of the covariances is estimated in a standard multivariate
    fashion. However, the individual row- and column covariances can not be restored as the
    Kronecker product is not revertable.
    """

    def __init__(self, dim, estimator='EmpiricalCovariance', init_mean_val=0, init_var_val=1):
        """
        TODO
        """
        self.Mu = init_mean_val * np.ones(dim)
        self.Mu_vec = self.Mu.flatten(order='F').reshape((dim[0]*dim[1], 1))
        self.Sigma_kron = init_var_val * np.kron(np.eye(dim[1]), np.eye(dim[0]))
        self._id_rc = np.eye(dim[0]*dim[1])
        # Setup estimator
        if estimator == 'EmpiricalCovariance':
            self._estimator = covariance.EmpiricalCovariance()
        elif estimator == 'LedoitWolf':
            self._estimator = covariance.LedoitWolf()
        elif estimator == 'MinCovDet':
            self._estimator = covariance.MinCovDet()
        elif estimator == 'OAS':
            self._estimator = covariance.OAS()
        elif estimator == 'ShrunkCovariance':
            self._estimator = covariance.ShrunkCovariance()
        else:
            raise ValueError('Unknown estimator: {}'.format(estimator))

    def update_params(self, samples):
        """
        TODO
        (estimate according to: AN EXPECTATION-MAXIMIZATION ALGORITHM FOR THE MATRIX
        NORMAL DISTRIBUTION,  HUNTER GLANZ AND LUIS CARVALHO)
        """
        self.Mu = self.estimate_mean_vec(samples)
        self.Mu_vec = self.Mu.flatten(order='F').reshape(self.Mu_vec.shape)
        samples_vec = [samp.flatten(order='F').reshape(self.Mu_vec.shape) for samp in samples]
        self.Sigma_kron = self.estimate_cov_kron(samples_vec, self.Mu_vec)

    def diff(self, other):
        """
        TODO
        """
        if not isinstance(other, MatrixGaussianKronParams):
            raise ValueError('Given instance is not of type {}'.format(type(self)))
        d1 = np.abs(self.Mu_vec - other.Mu_vec)
        d2 = np.abs(self.Sigma_kron - other.Sigma_kron)
        return np.sum(d1) + np.sum(d2)

    def estimate_mean_vec(self, samples):
        """
        TODO
        """
        mu = np.zeros(samples[0].shape)
        for samp in samples:
            mu = mu + samp
        return (1.0/len(samples))*mu

    def estimate_cov_kron(self, samples, mean):
        """
        TODO
        each samples: p x q
        cov_r: p x p
        cov_c: q x q
        """
        self._estimator.fit(np.squeeze(np.array(samples)))
        return self._estimator.covariance_
