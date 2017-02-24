#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Foobar."""

import numpy as np
from pymtl.interfaces.mtl_matrix_priors import MatrixGaussianParams, MatrixGaussianKronParams
from pymtl.interfaces.mtl_bayesian_prior_models import BayesPriorTL

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class BayesKroneckerRegression(BayesPriorTL):
    """Regression using matrix-variate Gaussian prior and matrix weight vector through Kronecker
    product formulations."""

    def __init__(self, max_prior_iter=1000, prior_conv_tol=1e-4, lam=1, lam_style=None,
                 use_kron_prior=True):
        """
        TODO
        """
        super(BayesKroneckerRegression, self).__init__(max_prior_iter, prior_conv_tol, lam, lam_style)
        self._prior = None
        self._weights = None
        self.use_kron_prior = use_kron_prior

    def fit(self, features, targets):
        """Foobar.

        Args:
            X (np.ndarray):
             X is ``(#trials x #dimensions)``.

            Y (np.ndarray):
             Y is ``(#trials x #dimensions)``.
        """
        assert features.shape[0] == targets.shape[0]
        # Setup prior if not already done
        if self._prior is None:
            self.init_model(features.shape, targets.shape)

        n_samples = features.shape[0]
        dim_k = targets.shape[1]
        dim_m = features.shape[1]
        # Compute Kronecker features
        kron_features = np.empty((n_samples, dim_k, dim_k*dim_m))
        id_k = np.eye(dim_k)
        for i in range(n_samples):
            kron_features[i, :, :] = np.kron(features[i], id_k)
        # Compute sum terms
        NN = np.zeros((dim_k*dim_m, 1))
        Ny = np.zeros((dim_k*dim_m, 1))
        for i in range(n_samples):
            NN = NN + kron_features[i].T.dot(kron_features[i])
            Ny = Ny + kron_features[i].T.dot(targets[i].reshape((dim_k, 1)))
        # Mean result
        NN = NN / float(n_samples)
        Ny = Ny / float(n_samples)
        # Perform linear regression MAP estimate on vectorized weights
        if isinstance(self._prior, MatrixGaussianKronParams):
            Sigma_kron = self._prior.Sigma_kron
            vec_Mu = self._prior.Mu_vec
        else:
            Sigma_kron = np.kron(self._prior.Sigma_c, self._prior.Sigma_r)
            vec_Mu = self._prior.Mu.flatten(order='F').reshape((dim_k*dim_m, 1))
        self._weights = np.linalg.lstsq(
            Sigma_kron.dot(NN) + self.lam*np.eye(dim_k*dim_m),
            Sigma_kron.dot(Ny) + self.lam*vec_Mu)[0]
        # Unvectorize weights
        self._weights = self._weights.reshape((dim_k, dim_m), order='F')
        return self

    def predict(self, features):
        """
        TODO
        Expected: Features in form (#n_samples, #n_feature)
        Output: Predictions in form (#n_sampels, #n_outputs)
        """
        if self._weights is None:
            W = self._prior.Mu
        else:
            W = self._weights
        assert features.shape[1] == W.shape[1], 'Feature dimension mismatch!'
        # Perform predictions
        pred = features.dot(W.T)
        return pred

    def score(self, features, targets):
        """
        TODO
        Expected: Features in form (#n_samples, #n_feature)
                  Targets in form (#n_sampels, #n_outputs)
        """
        score = self.loss(features, targets)
        return score

    def loss(self, features, targets):
        """
        TODO
        """
        err = np.linalg.norm(targets - self.predict(features)) ** 2
        return err/len(features)

    def init_model(self, dim_features, dim_targets, init_val=0):
        """
        TODO
        """
        if self.use_kron_prior:
            prior = MatrixGaussianKronParams((dim_targets[1], dim_features[1]),
                                             estimator='EmpiricalCovariance', init_mean_val=init_val, init_var_val=1)
        else:
            prior = MatrixGaussianParams((dim_targets[1], dim_features[1]), init_mean_val=init_val, init_var_val=1)
        self.set_prior(prior)
        self._weights = np.copy(self._prior.Mu)

    def get_weights(self):
        """
        TODO
        """
        if self._weights is not None:
            return self._weights
        else:
            return self._prior.Mu

    def set_weights(self, weights):
        """
        TODO
        """
        if weights is None:
            self._weights = None
        else:
            self._weights = np.copy(weights)

    def get_prior(self):
        """
        TODO
        """
        return self._prior

    def set_prior(self, prior):
        """
        TODO
        """
        self._prior = prior
