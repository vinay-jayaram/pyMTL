#!/usr/bin/env python

import numpy as np
from sklearn import metrics
from sklearn import covariance
from pymtl.interfaces.mtl_bayesian_prior_models import BayesPriorTL
from pymtl.misc import numerics

__author__ = "Karl-Heinz Fiebig, Vinay Jayaram"
__copyright__ = "Copyright 2017"

class SpatioTemporalBayesRegression(BayesPriorTL):
    """
    TODO
    """

    def __init__(self, max_prior_iter=1000, prior_conv_tol=1e-6, lam=1, lam_style='ML', target_rank=8):
        """
        TODO
        lam_style routines are not implemented yet
        """
        # Associate settings
        self._lam_style = lam_style
        self.max_prior_iter=max_prior_iter
        self.prior_conv_tol = prior_conv_tol
        self.lam = lam
        self.target_rank = target_rank
        # Setup internala structures
        self._prior_mu = None
        self._prior_Sigma = None
        self._weights = None
        self._cov_estimator = covariance.OAS()

    def fit_multi_task(self, lst_X, lst_y):
        """
        TODO
        Expects list of task data sets in format trials x channels x samples and associated
        labels
        """
        n_tasks = len(lst_X)
        # re-initialize model
        self.init_model(lst_X[0].shape[1:])
        for iter in range(self.max_prior_iter):
            old_weights = self._weights
            lst_weights = []
            for task_idx in range(n_tasks):
                X = lst_X[task_idx]
                y = lst_y[task_idx]
                self.fit(X, y)
                lst_weights.append(np.copy(self._weights))
            # Update prior and set new weights
            self._update_prior(lst_weights)
            self._weights = self._prior_mu
            # Compute prior errors, weight matrix tank and convergence status
            errors = [self.loss(lst_X[task_idx], lst_y[task_idx]) for task_idx in range(n_tasks)]
            rank = np.linalg.matrix_rank(numerics.unvech(self._prior_mu, norm_conserved=False))
            conv = np.sum(np.abs(self._weights - old_weights) < self.prior_conv_tol)
            print('[Iteration {}/{}] Mean prior loss: {}; prior mean rank: {}; converged: {}/{}'.format(
                iter+1, self.max_prior_iter, np.mean(errors), rank, conv, len(self._weights)))
            if conv == len(self._weights):
                print('MTL converged!')
                break


    def fit(self, X, y):
        """
        TODO
        Expects feature format X: (trials, channels, samples) and format label y: (trials,)
        """
        # Check if model has to be initialized
        if self._weights is None:
            self.init_model(X.shape[1:])
        # Compute to half-vectorized covariance features and regress labels
        X = self._to_cov_features(X)
        y = y.flatten()[:, np.newaxis]
        covX = self._prior_Sigma.dot(X.T)
        self._weights = np.linalg.lstsq(1.0/self.lam * covX.dot(X) + np.eye(X.shape[1]),
                                        1.0/self.lam * covX.dot(y) + self._prior_mu)[0]
        #self._weights = self._reduce_weight_rank(self._weights)

    def loss(self, X, y):
        """
        TODO
        Squared loss
        """
        y_pred = self.predict(X)
        diff = (y_pred - y)[:, np.newaxis]
        loss = diff.T.dot(diff)[0, 0]
        #reg = self.lam * self._weights.T.dot(self._weights)[0, 0]
        return loss #+ reg

    def predict(self, X):
        """
        TODO
        Raw regression predicion (no classification!)
        """
        # Check if model has to be initialized
        if self._weights is None:
            self.init_model(X.shape[1:])
        # Compute to half-vectorized covariance features and regress labels
        X = self._to_cov_features(X)
        return X.dot(self._weights).flatten()

    def score(self,X, y):
        return np.mean(np.sign(self.predict(X)) == y)

    def init_model(self, data_dim):
        d = data_dim[0]
        self._prior_mu = np.zeros((int(0.5*(d*d+d)), 1))
        self._prior_Sigma = np.eye(len(self._prior_mu))
        self._weights = self._prior_mu

    def _to_cov_features(self, X):
        n_samples = X.shape[0]
        features = np.empty((n_samples, len(self._weights)))
        for idx in range(n_samples):
            X_s = X[idx, :]
            #self._cov_estimator.fit(X_s.T)
            #cov = self._cov_estimator.covariance_
            cov = X_s.dot(X_s.T)
            features[idx, :] = numerics.vech(cov, conserve_norm=False)
        return features

    def _update_prior(self, lst_weights):
        weight_matrix = np.array(lst_weights).squeeze()
        # Estimate mean
        self._prior_mu = np.mean(weight_matrix, axis=0)[:, np.newaxis]
        # Reduce rank of weights to desired rank
        self._prior_mu = self._reduce_weight_rank(self._prior_mu)
        # Remove mean from weight matrix and estimate covariance
        # Estimate covariance
        self._cov_estimator.fit(weight_matrix)
        self._prior_Sigma = self._cov_estimator.covariance_

    def _reduce_weight_rank(self, w):
        # Unvectorize and approximate weights with low-rank matrix
        W = numerics.unvech(w, norm_conserved=False)
        lam = 0.0
        step_size = 0.1
        Z = W
        while np.linalg.matrix_rank(Z) > self.target_rank and lam <= 1.0:
            Z = numerics.complete_matrix(W, lam=lam, beta=1e100)
            lam = lam + step_size
            print(np.linalg.matrix_rank(Z))
        # print if Z is symmetric
        print('Symmetric: {}'.format((Z ==Z.T).all()))
        # Vectorize weights again and return them
        return numerics.vech(Z, conserve_norm=False)[:, np.newaxis]


