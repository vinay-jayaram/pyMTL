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

