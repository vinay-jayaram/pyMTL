# /usr/bin/env python
# -*- coding: utf-8 -*-

"""mtl_gaussian_priors_models.py: Interface to multi-task transfer learning models."""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import numpy as np
from abc import ABCMeta, abstractmethod
from pymtl.interfaces.mtl_priors import PriorParamsInterface
from pymtl.interfaces.mtl_base import TransferLearningBase
from pymtl.misc import verbose as vb
import time
import copy

__author__ = "Vinay Jayaram, Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"

class FDGradientInterface(object):
    """
    Abstract class interface to retrieve the gradient for feature decomposition
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_bilinear_gradient(self, features, targets, sec_weights):
        pass

class BayesPriorTL(TransferLearningBase):
    '''
    General class for models that rely on a Gaussian prior to update some sort of
    regularization parameter that can be trained separately for each of the
    datasets.
    '''
    __metaclass__ = ABCMeta

    def multi_task_models(self):
        """
        Returns a list of models that were fit to the individual datasets in the process of
        multi-task training.
        """
        return self._task_models

    def new_task_model(self):
        """
        Returns the task-adapted model.
        """
        return self

    def __init__(self, max_prior_iter, prior_conv_tol, lam, lam_style):
        """Constructor for an instance of GaussianPriorTL.

        Abstract class constructor to setup the parameters for multi-task training of a
        multivariate Gaussian prior over the inheriting' model weights.

        Args:
            max_prior_iter (int):
                The maximum number of iterations to perform for prior training. If convergence is
                not reached after the maximum number of iterations, a warning will be displayed and
                prior training terminates with the current state.

            prior_conv_tol (float):
                The tolerance at which the prior is considered to be converged. Convergence occurs
                if there is no significant difference (as specified by the tolerance value) between
                two consecutive iterations.

            lam (float):
                The lambda value to trade-off between deviation from the prior and fitting
                task-specific structure.
        """
        self.max_prior_iter = max_prior_iter
        self.prior_conv_tol = prior_conv_tol
        self.lam = lam
        self.lam_style = lam_style
        self._attr_prior = None
        self._attr_weights = None

        # Init other attributes
        self._task_models = []
        self._adapted_model = None
        self._num_iters = 0

    def fit_multi_task(self, lst_features, lst_targets, verbose=vb.ON):
        """
        Generic fitting of all tasks TODO
        """
        n_tasks = len(lst_features)
        if (n_tasks != len(lst_targets)):
            raise ValueError('Number of tasks for features and targets does not match.')
        if (n_tasks == 0):
            raise ValueError('Number of tasks has to be geater than zero')
        dim_features = lst_features[0].shape
        dim_targets = lst_targets[0].shape

        # Init prior and create a model for each task dataset
        self.init_model(dim_features, dim_targets, init_val=0)
        self._task_models = [] # Reset models already stored in this instance before cloning
        self._task_models = [self.clone() for i in range(n_tasks)]
        # Start prior training
        it = 0
        print(self.max_prior_iter)
        for it in range(self.max_prior_iter):
            prev_prior = copy.deepcopy(self.prior)
            # Train task-specific models
            lst_weights = []
            lst_scores = []
            lst_loss = []
            start = time.time()
            for idx, model in enumerate(self._task_models):
                # Train model with current prior
                model.prior = self.prior
                model.fit(lst_features[idx], lst_targets[idx])
                # Gather results in the appropriate list
                lst_weights.append(model.weights)
                lst_scores.append(model.score(lst_features[idx], lst_targets[idx]))
                lst_loss.append(model.loss(lst_features[idx], lst_targets[idx]))
            # Update priors in this model from task-specific weights
            prior = self.prior
            diff = 0
            if isinstance(prior, list):
                for p_idx in range(len(prior)):
                    if not isinstance(prior[p_idx], PriorParamsInterface):
                        raise ValueError('Given instance is not of type PriorParamsInterface')
                    prior[p_idx].update_params([weights[p_idx] for weights in lst_weights])
                    diff += prior[p_idx].diff(prev_prior[p_idx])
            else:
                if not isinstance(prior, PriorParamsInterface):
                    raise ValueError('Given instance is not of type PriorParamsInterface')
                prior.update_params(lst_weights)
                #import pdb; pdb.set_trace()
                diff += prior.diff(prev_prior)
            # Update lambda value according to desired method
            if self.lam_style == 'ML':
                # Maximum-Likelihood estimate
                # TODO is this a general ML estimate or does this apply only to linear regression?
                lam = np.sum([len(X) for X in lst_features]) / (2*np.sum(lst_loss))
                self.set_params(lam=lam)
                for model in self._task_models:
                    model.set_params(lam=lam)
            end = time.time()
            vb.pyout('[{}] Prior Iteration {} ({}s); Convergence: {}; lambda: {}; mean loss: {}'.format(
                type(self).__name__, it, round(end - start, 1), diff, self.lam, str(round(np.mean(lst_loss), 4))
            ), lvl=verbose)
            if diff <= self.prior_conv_tol:
                break
        self.weights = None
        self.prior = prior
        self._num_iters = it
        return self

    ### Abstract interfaces to be implemented by inheriting models ###

    @abstractmethod
    def fit(self, features, targets):
        """
        Fits a model to the given data set and stores relevant parameters into self.weights.
        """
        pass

    @abstractmethod
    def predict(self, features):
        """
        Predicts the labels for the given features.
        """
        pass

    @abstractmethod
    def score(self, features, targets):
        """
        Computes a goodness of fit measure of this model on the given dataset.
        """
        pass

    @abstractmethod
    def loss(self, features, targets):
        """
        Computes a goodness of fit measure of this model on the given dataset.
        """
        pass

    @abstractmethod
    def init_model(self, dim_features, dim_targets, init_val):
        """
        Computes a goodness of fit measure of this model on the given dataset.
        """
        pass

    @property
    def weights(self):
        """
        Returns the weight parameters used by this model.
        """
        return self._attr_weights

    @weights.setter
    def weights(self, newweights):
        """
        Returns the weight parameters used by this model.
        """
        self._attr_weights = newweights

    @property
    def prior(self):
        """
        Returns the priors used by the model.
        """
        return self._attr_prior

    @prior.setter
    def prior(self, newprior):
        """
        Sets the priors used by the model.
        """
        self._attr_prior = newprior
