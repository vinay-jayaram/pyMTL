#!/usr/bin/env python

import numpy as np
from pymtl.interfaces.mtl_bayesian_prior_models import BayesPriorTL
import pymtl.interfaces.mtl_priors as priors
from sklearn import metrics
from pymtl.misc import numerics 

__author__ = "Vinay Jayaram, Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class BayesRegression(BayesPriorTL):
    """
    Implements standard L2-loss linear regression with optional prior learning.
    """

    def __init__(self, max_prior_iter=1000, prior_conv_tol=1e-4, lam=1, 
                 lam_style='ML', estimator='OAS'):
        """
        max_prior_iter: see mtl_bayesian_prior_models
        prior_conv_tol: see mtl_bayesian_prior_models
        lam:            see mtl_bayesian_prior_models
        lam_style:      see mtl_bayesian_prior_models
        """
        super(BayesRegression, self).__init__(max_prior_iter, prior_conv_tol, lam, lam_style)
        self._classes = None
        self.estimator = estimator

    def fit(self, features, targets, lam=None):
        """
        Computes standard linear regression solution given current prior. 
        """
        # data safety
        if features.shape[0] != targets.shape[0]:
            raise ValueError('Number of samples in data set ({}) does not match number of \
                             samples ({}) in the target vector'.format(features.shape[0],
                                                                       targets.shape[0]))
        X_train = features

        y_train = targets.reshape(len(targets), 1)
        
        if lam is None:
            lam = self.lam
        # Setup prior if not already done
        if self.prior is None:
            self.init_model(X_train.shape, y_train.shape)
        covX = self.prior.Sigma.dot(X_train.T)
        self.weights = np.linalg.lstsq(1.0/lam*covX.dot(X_train) + np.eye(X_train.shape[1]),
                                        (1.0/lam*covX.dot(y_train)) + self.prior.mu)[0]
        return self


    def predict(self, features):
        """
        Returns predicted values given features
        """
        # TODO arg checks
        if self.weights is None:
            w = self.prior.mu
        else:
            w = self.weights
        pred = features.dot(w)
        return pred

    def score(self, features, targets):
        """
        If classifier, returns accuracy score on given samples and labels. If not, returns loss
        """
        score = self.loss(features, targets)
        return score

    def loss(self, features, targets):
        """
        Specifies squared loss for this particular model
        """
        X = features
        y = targets.reshape(len(targets), 1)
        if self.weights is None:
            w = self.prior.mu
        else:
            w = self.weights
        pred = X.dot(w)
        err = np.sum(np.power(y-pred, 2)) #/ len(y)
        return err

    def init_model(self, dim, dim_targets, init_val=0):
        """
        Initialize the prior given an initial value
        """
        prior = priors.SKGaussianParams(dim[1], estimator=self.estimator, 
                                 init_mean_val=init_val, init_var_val=1)
        self.prior = prior
        self.weights = np.copy(self.prior.mu)

    @BayesPriorTL.weights.getter
    def weights(self):
        """
        TODO
        """
        if self.weights is not None:
            return self._attr_weights
        else:
            return self.prior.mu

    @BayesPriorTL.weights.setter
    def weights(self, weights):
        """
        TODO
        """
        if weights is None:
            self._attr_weights = None
        else:
            self._attr_weights = np.copy(weights)

class BayesRegressionClassifier(BayesRegression):

    def __init__(self, max_prior_iter=100, prior_conv_tol=1e-4, lam=1, 
                 lam_style='ML', estimator='OAS'):
        """
        is_classifier:  converts to internal label representation if true
        max_prior_iter: see mtl_bayesian_prior_models
        prior_conv_tol: see mtl_bayesian_prior_models
        lam:            see mtl_bayesian_prior_models
        lam_style:      see mtl_bayesian_prior_models
        TODO: Allow for non {-1,1} internal labelling
        """
        self._set_internal_classes([-1,1])
        super(BayesRegressionClassifier, self).__init__(max_prior_iter, prior_conv_tol, lam, 
                                                        lam_style, estimator)


    def fit(self, features, targets):
        """
        Computes standard linear regression solution given current prior. Switches labels
        to allow for classification
        """
        y_train, self._classes = self._convert_classes(targets)
        super(BayesRegressionClassifier, self).fit(features, y_train)
        return self

    def _convert_classes(self, targets):
        # Exract classes and save them as {0, 1} targets
        classes, inv = np.unique(targets, return_inverse=True)
        if len(classes) != 2:
            raise ValueError('Expected exactly two classes for binary classification, but got {}'.format(len(classes)))
        # Convert to {0, 1} targets
        y = inv.reshape(len(inv), 1)
        for ind in range(2):
            y[inv==ind] = self._internal_classes[ind]

        return y, classes

    def _set_internal_classes(self, classes):
        self._internal_classes = classes

    def _recover_classes(self, targets):
        # Cast class labels back to the original classes
        y = np.copy(targets)
        class_inds = []
        for ind in range(2):
            class_inds.append(np.where(targets == self._internal_classes[ind]))
        for ind in range(2):
            y[class_inds[ind]] = ind
        return np.array([self._classes[int(i)] for i in y])

    def loss(self, features, targets):
        """
        Specifies squared loss for this particular model
        """
        y = self._convert_classes(targets)[0]
        return super(BayesRegressionClassifier,self).loss(features, y)

    def predict(self, features):
        """
        Returns predicted values given features
        """
        pred = super(BayesRegressionClassifier,self).predict(features)
        pred = self._recover_classes(np.sign(pred))
        return pred

    def score(self, features, targets):
        """
        If classifier, returns accuracy score on given samples and labels. If not, returns loss
        """
        
        return metrics.accuracy_score(self.predict(features), targets.flatten())

class SpatiotemporalRegressionClassifier(BayesRegressionClassifier):
    
    def __init__(self, nu=1, prior_rank=5, max_prior_conv_iter=10, max_prior_iter=100,
                 prior_conv_tol=1e-4, lam=1, lam_style='ML', estimator='OAS'):
        """
        is_classifier:  converts to internal label representation if true
        max_prior_iter: see mtl_bayesian_prior_models
        prior_conv_tol: see mtl_bayesian_prior_models
        lam:            see mtl_bayesian_prior_models
        lam_style:      see mtl_bayesian_prior_models
        TODO: Allow for non {-1,1} internal labelling
        """
        super(SpatiotemporalRegressionClassifier, self).__init__(max_prior_iter,
                                                                 prior_conv_tol, 
                                                                 lam,
                                                                 lam_style,
                                                                 estimator)
        self.nu = nu
        self.prior_rank = prior_rank
        self.max_prior_conv_iter = max_prior_conv_iter

    def fit(self, features, targets):
        if features.ndim == 3:
            features = self._to_cov_features(features)
        super(SpatiotemporalRegressionClassifier, self).fit(features, targets)

    def _to_cov_features(self, X):
        n_samples = X.shape[0]
        if self.weights is None:
            self.init_model(X.shape)
        features = np.empty((n_samples, len(self.weights)))
        for idx in range(n_samples):
            X_s = np.squeeze(X[idx,...])
            #self._cov_estimator.fit(X_s.T)
            #cov = self._cov_estimator.covariance_
            cov = X_s.dot(X_s.T) / (n_samples-1)
            features[idx, :] = numerics.vech(cov)
        return features

    def init_model(self, dim, dim_targets=0, init_val=0):
        vech_dim = dim[1]*(dim[1]+1)/2
        #prior = LowRankGaussianParams(vech_dim, nu=self.nu, 
        #                              estimator=self.estimator, 
        #                              k=self.prior_rank, 
        #                              max_its=self.max_prior_conv_iter)
        prior = priors.SKGaussianParams(vech_dim, estimator=self.estimator,
                                 init_mean_val=init_val, init_var_val=1)
        self.prior = prior
        self.weights = np.copy(self.prior.mu)

    def predict(self,features):
        if features.ndim == 3:
            features = self._to_cov_features(features)
        return super(SpatiotemporalRegressionClassifier, self).predict(features)


    def loss(self, features, targets):
        if features.ndim == 3:
            features = self._to_cov_features(features)
        return super(SpatiotemporalRegressionClassifier, self).loss(features, targets)

class TemporalBRC(BayesRegressionClassifier):
    
    def __init__(self,max_prior_iter=100, prior_conv_tol=1e-4, lam=1,
                 lam_style='ML', estimator='OAS'):
        super(TemporalBRC, self).__init__(max_prior_iter, prior_conv_tol, lam,
                                          lam_style, estimator)

    def init_model(self, dim, dim_targets, init_val=0):
        prior = priors.SKGaussianParams(dim[1], estimator=self.estimator, init_mean_val=init_val,
                           init_var_val=1)
        self.prior = prior
        self.weights = np.copy(self.prior.mu)
