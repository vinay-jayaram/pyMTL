#!/usr/bin/env python

import numpy as np
from pymtl.interfaces.mtl_bayesian_prior_models import BayesPriorTL
import pymtl.interfaces.mtl_priors as priors
from sklearn import metrics


__author__ = "Vinay Jayaram, Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class BayesRegression(BayesPriorTL):
    """
    Implements standard L2-loss linear regression with optional prior learning.
    """

    def __init__(self, max_prior_iter=1000, prior_conv_tol=1e-4, lam=1, 
                 lam_style='ML', estimator='OAS', parallel=False):
        """
        max_prior_iter: see mtl_bayesian_prior_models
        prior_conv_tol: see mtl_bayesian_prior_models
        lam:            see mtl_bayesian_prior_models
        lam_style:      see mtl_bayesian_prior_models
        """
        super(BayesRegression, self).__init__(max_prior_iter, prior_conv_tol, lam, lam_style, parallel)
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
        pred = features.dot(self.weights).flatten()
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
        y = targets.flatten()
        pred = self.predict(X)
        #import pdb; pdb.set_trace()
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
                 lam_style='ML', estimator='OAS', parallel=False):
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
                                                        lam_style, estimator, parallel)


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

    def predict_raw(self, features):
        """
        Get raw predicted values
        """
        return super(BayesRegressionClassifier, self).predict(features)

    def predict(self, features):
        """
        Returns predicted values given features
        """
        return self._recover_classes(np.sign(self.predict_raw(features))).flatten()

    def score(self, features, targets):
        """
        If classifier, returns accuracy score on given samples and labels. If not, returns loss
        """
        
        return metrics.accuracy_score(self.predict(features), targets.flatten())

    def tests(self, X, Y):
        '''
        (X,Y): takes data and performs sanity checks to ensure there are no silly errors

        Current tests:
        1. Ensure there are members of both classes in the output (overfit, just to 
        confirm that it trains)
        2. print the means of the classes in the projected space
        '''

        # test 1
        # train model on data
        self.fit(X,Y)
        yhat = self.predict(X)
        vals = self.predict_raw(X)
        u = np.unique(Y)
        print('Score on training data: {}'.format(self.score(X,Y)))
        print('Class 0 correct: {}/{}'.format((yhat == u[0]).sum(),(Y==u[0]).sum()))
        print('Projected mean of class 0: {:.2f}'.format(self.predict_raw(X[Y==u[0],...]).mean()))
        print('Class 1 correct: {}/{}'.format((yhat == u[1]).sum(),(Y==u[1]).sum()))
        print('Projected mean of class 1: {:.2f}'.format(self.predict_raw(X[Y==u[1],...]).mean()))
