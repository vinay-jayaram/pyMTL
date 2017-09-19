# /usr/bin/env python
# -*- coding: utf-8 -*-
"""mtl_gaussian_priors_models.py: Interface to multi-task transfer learning models."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from abc import ABCMeta, abstractmethod
from pymtl.interfaces.priors import PriorParamsInterface
from pymtl.interfaces.base import TransferLearningBase
from pymtl.misc import verbose as vb
from joblib import Parallel, delayed
import multiprocessing
import time
import copy

__author__ = "Vinay Jayaram, Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


def single_fit(model, X, Y, prior):
    model.prior = prior
    model.fit(X, Y)
    return model.weights, model.score(X, Y), model.loss(X, Y)


class FDGradientInterface(object):
    """
    Abstract class interface to retrieve the gradient for feature decomposition
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_bilinear_gradient(self, features, targets, sec_weights):
        pass


class BayesMTL(TransferLearningBase):
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

    def __init__(self, max_prior_iter, prior_conv_tol, C, C_style, prior=None):
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

            C (float):
                The lambda value to trade-off between deviation from the prior and fitting
                task-specific structure.

            C_style (str: 'ML' or ''):
                Whether to use ML lambda estimation or leave it fixed
        """
        self.max_prior_iter = max_prior_iter
        self.prior_conv_tol = prior_conv_tol
        self.C = C
        self.C_style = C_style
        self._attr_prior = prior
        self._attr_weights = None

        # Init other attributes
        self._task_models = []
        self._adapted_model = None
        self._num_iters = 0

    def pre_fit(self, lst_features, lst_targets, verbose=vb.WARN, n_jobs=1):
        self.fit_multi_task(lst_features, lst_targets, verbose, n_jobs)
        return self

    def fit_multi_task(self, lst_features, lst_targets, verbose, n_jobs=1):
        """
        Generic fitting of all tasks TODO
        """
        n_tasks = len(lst_features)
        if (n_tasks != len(lst_targets)):
            raise ValueError(
                'Number of tasks for features and targets does not match.')
        if (n_tasks == 0):
            raise ValueError('Number of tasks has to be geater than zero')
        dim_features = lst_features[0].shape
        dim_targets = lst_targets[0].shape

        # Init prior and create a model for each task dataset
        self.init_model(dim_features, dim_targets, init_val=0)
        self._task_models = [
        ]  # Reset models already stored in this instance before cloning
        self._task_models = [self.clone() for i in range(n_tasks)]
        if n_jobs == 1:
            print('Using serial approach to task updates')
            self._multitask_update(lst_features, lst_targets, verbose, 1)
        elif n_jobs == None:
            print(
                'Automatically determining cores to process with. Attempting parallel processing with {} cores'.
                format(multiprocessing.cpu_count() - 1))
            self._multitask_update(lst_features, lst_targets, verbose,
                                            multiprocessing.cpu_count() - 1)
        else:
            print(
                'Attempting parallel processing with {} cores'.format(n_jobs))
            self._multitask_update(lst_features, lst_targets, verbose,
                                            n_jobs)
        return self

    def _multitask_update(self, lst_features, lst_targets, verbose,
                                   ncores):
        '''
        Multitask parallel update if appropriate thing is tweaked. 
        '''
        with Parallel(n_jobs=ncores) as PAR:
            it = 0
            while it < self.max_prior_iter:
                prev_prior = copy.deepcopy(self.prior)
                # Train task-specific models
                lst_weights = []
                lst_scores = []
                lst_loss = []
                start = time.time()
                par_out = PAR(
                    delayed(single_fit)(self._task_models[i], lst_features[i],
                                        lst_targets[i], self.prior)
                    for i in range(len(self._task_models)))
                for w, s, l in par_out:
                    lst_weights.append(w)
                    lst_scores.append(s)
                    lst_loss.append(l)
                # Update priors in this model from task-specific weights
                prior = self.prior
                diff = 0
                if isinstance(prior, list):
                    for p_idx in range(len(prior)):
                        if not isinstance(prior[p_idx], PriorParamsInterface):
                            raise ValueError(
                                'Given instance is not of type PriorParamsInterface'
                            )
                        prior[p_idx].update_params(
                            [weights[p_idx] for weights in lst_weights])
                        diff += prior[p_idx].diff(prev_prior[p_idx])
                else:
                    if not isinstance(prior, PriorParamsInterface):
                        raise ValueError(
                            'Given instance is not of type PriorParamsInterface'
                        )
                    prior.update_params(lst_weights)
                    #import pdb; pdb.set_trace()
                    diff += prior.diff(prev_prior)
                # Update lambda value according to desired method
                if self.C_style == 'ML':
                    # Maximum-Likelihood estimate
                    # TODO is this a general ML estimate or does this apply only to linear regression?
                    C = np.sum([len(X) for X in lst_features]) / (
                        2 * np.sum(lst_loss))
                    self.set_params(C=C)
                    for model in self._task_models:
                        model.set_params(C=C)
                end = time.time()
                vb.pyout(
                    '[{}] Prior Iteration {} ({:.2e}s); Convergence: {:.2e}; lambda: {:.2e}; mean loss: {:.2e}'.
                    format(
                        type(self).__name__, it,
                        round(end - start, 1), diff, self.C,
                        np.mean(lst_loss)),
                    lvl=verbose)
                if diff <= self.prior_conv_tol:
                    break
                it += 1
        self.weights = None
        self.prior = prior
        self._num_iters = it

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
