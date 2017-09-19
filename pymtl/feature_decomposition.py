#!/usr/bin/env python

import numpy as np
from pymtl.interfaces.models import BayesMTL
from pymtl.interfaces.gradient_interface import GradientInterface
from pymtl.linear_regression import MTLRegression
from pymtl.logistic_regression import MTLLogisticRegression

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class FeatureDecompositionModel(BayesMTL):
    """
    TODO
    """

    def __init__(self,
                 max_prior_iter=1000,
                 prior_conv_tol=1e-4,
                 C=1,
                 C_style=None,
                 Type='linear',
                 max_fd_iter=100,
                 fd_conv_tol=1e-3):
        """
        TODO
        """
        super(FeatureDecompositionModel, self).__init__(
            max_prior_iter, prior_conv_tol, C, C_style)
        self.Type = Type
        if Type == 'linear':
            self.spatial_model = BayesRegressionClassifier(
                max_prior_iter=max_prior_iter,
                prior_conv_tol=prior_conv_tol,
                C=C,
                C_style=C_style)
            self.spectral_model = self.spatial_model.clone()
        elif Type == 'logistic':
            self.spatial_model = BayesLogisticClassifier(
                max_prior_iter=max_prior_iter,
                prior_conv_tol=prior_conv_tol,
                C=C,
                C_style=C_style,
                optim_algo='gd',
                pred_threshold=0.5)
            self.spectral_model = self.spatial_model.clone()
        else:
            raise ValueError('Given type {} unknown'.format(Type))
        self.max_fd_iter = max_fd_iter
        self.fd_conv_tol = fd_conv_tol

    def fit(self, features, targets):
        """
        TODO
        """
        # data safety
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                'Number of samples in data set ({}) does not match number of \
                             samples ({}) in the target vector'
                .format(features.shape[0], targets.shape[0]))
        # Setup prior if not already done
        if self.get_prior()[0] is None or self.get_prior()[1] is None:
            self.init_model(features.shape, targets.shape)
        # Setup current parameters
        self.spatial_model.set_params(C=self.C)
        self.spectral_model.set_params(C=self.C)

        if isinstance(self.spatial_model, GradientInterface) and isinstance(
                self.spectral_model, GradientInterface):
            self._fit_gradient_based(features, targets)
        else:
            self._fit_model_based(features, targets)
        return self

    def predict(self, features):
        """
        TODO
        """
        Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
        return self.spatial_model.predict(Xw)

    def predict_proba(self, features):
        """
        TODO
        """
        Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
        return self.spatial_model.predict_proba(Xw)

    def predict_log_proba(self, features):
        """Log of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self._classes``.
        """
        Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
        return self.spatial_model.predict_log_proba(Xw)

    def score(self, features, targets):
        """
        TODO
        """
        # Setup current parameters
        self.spatial_model.set_params(C=self.C)
        self.spectral_model.set_params(C=self.C)
        Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
        return self.spatial_model.score(Xw, targets)

    def loss(self, features, targets):
        """
        TODO
        """
        # Setup current parameters
        self.spatial_model.C = self.C
        self.spectral_model.C = self.C
        # We will just use the error of the spatial model as it depends on the spectral weights
        Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
        spat_err = self.spatial_model.loss(Xw, targets)
        return spat_err

    def init_model(self, dim, dim_targets, init_val=0):
        """
        TODO
        """
        self.spatial_model.init_model(
            (dim[0], dim[1]), dim_targets, init_val=1.0 / np.sqrt(dim[1]))
        self.spectral_model.init_model(
            (dim[0], dim[2]), dim_targets, init_val=init_val)

    def get_weights(self):
        """
        TODO
        """
        return [
            self.spatial_model.get_weights(),
            self.spectral_model.get_weights()
        ]

    def set_weights(self, weights):
        """
        TODO
        """
        if weights is None:
            self.spatial_model.set_weights(None)
            self.spectral_model.set_weights(None)
        else:
            assert (len(weights) == 2)
            self.spatial_model.set_weights(weights[0])
            self.spectral_model.set_weights(weights[1])

    def get_prior(self):
        """
        TODO
        """
        return [
            self.spatial_model.get_prior(),
            self.spectral_model.get_prior()
        ]

    def set_prior(self, prior):
        """
        TODO
        """
        assert (len(prior) == 2)
        self.spatial_model.set_prior(prior[0])
        self.spectral_model.set_prior(prior[1])

    def _fit_model_based(self, features, targets):
        loss_current = -999
        for iter in range(self.max_fd_iter):
            loss_prev = loss_current
            # Train spatial and spectral models with corresponding feature spaces
            Xa = np.squeeze(
                features.transpose(0, 2, 1).dot(
                    self.spatial_model.get_weights()))
            self.spectral_model.fit(Xa, targets)
            Xw = np.squeeze(features.dot(self.spectral_model.get_weights()))
            self.spatial_model.fit(Xw, targets)
            loss_current = self.loss(features, targets)
            if (np.abs(loss_prev - loss_current) <= self.fd_conv_tol):
                break

    def _fit_gradient_based(self, features, targets):
        """Minimizes the Cross-Entropy Loss objective w.r.t. the weights
        using gradient descent with adaptive global learning rate.

        The learning rate is adapted with additive increase and multiplicative
        decrease on detection of error increase (i.e. overshooting).

        Parameters
        ----------
        X : matrix, shape = (n_samples, k, d)
        y : vector, shape = (n_samples, 1)
        max_iter : integer, default 10000
        tol : float, default 1e-2
        verbose : {False, 'warn', 'enable'}, default 'warn'

        Returns
        -------
        a, w : shape=(k, 1), shape=(d, 1)
            The optimal spatial and spectral weights trained on the data set.
        """
        # Retrieve useful variables and setup internal parameters
        k = features.shape[1]
        d = features.shape[2]
        inc_rate = 0.1
        dec_rate = 0.025
        # Initialize weights and learning rates
        self.spectral_model.set_weights(None)
        self.spatial_model.set_weights(None)
        eta = 0.1
        ce_next = self.loss(features, targets)
        # Start gradient descent optimization
        for i in range(self.max_fd_iter):
            ce_current = ce_next
            w_current = self.spectral_model.get_weights()
            a_current = self.spatial_model.get_weights()
            # Update spectral weights
            Xa = np.squeeze(features.transpose(0, 2, 1).dot(a_current))
            grad_w = self.spectral_model.get_loss_gradient(Xa, targets)
            w_next = w_current - eta * grad_w
            self.spectral_model.set_weights(w_next)
            Xw = np.squeeze(features.dot(w_next))
            grad_a = self.spatial_model.get_loss_gradient(Xw, targets)
            a_next = a_current - eta * grad_a
            self.spatial_model.set_weights(a_next)
            # Check error status
            ce_next = self.loss(features, targets)
            # Adapt learning rate according to current error development
            if ce_next > ce_current:
                # Decrease learning rate and reject iteration
                eta = dec_rate * eta
                self.spectral_model.set_weights(w_current)
                self.spatial_model.set_weights(a_current)
                ce_next = ce_current
                continue
            else:
                # Slowly increase learning rate
                eta = eta + inc_rate * eta
            # Check convergence status
            diff_ce = np.abs(ce_next - ce_current)
            if diff_ce <= self.fd_conv_tol:
                break
