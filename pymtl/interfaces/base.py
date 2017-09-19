# /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn import base as skbase
import copy

__author__ = "Vinay Jayaram, Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class TransferLearningBase(skbase.BaseEstimator):
    """
    Base class to deal with transfer learning schemes in a consistent way. It is
    intended to provide a convenient wrapper to test different transfer learning
    approaches that require different sorts of constraints and inputs but need
    to be compared against each other.

    The class accepts datasets and related information and trains either one or
    many classifiers, each of which is an instantiation of the scikit-learn
    BaseEstimatior class. 

    """
    __metaclass__ = ABCMeta

    def clone(self, safe=True):
        #return skbase.clone(self, safe=safe)
        return copy.deepcopy(self)

    @abstractproperty
    def multi_task_models(self):
        """
        Returns a list of models that were fit to the individual datasets in the process of
        multi-task training.
        """
        pass

    @abstractproperty
    def new_task_model(self):
        """
        Returns the task-adapted model.
        """
        pass

    # def __init__(self):
    #     """
    #     Initialization for the algorithm. We follow scikit-learn convention here
    #     and require that initialize do no processing of inputs and simply adds
    #     them to a dictionary that each instance stores. Any data-dependent
    #     altering of hyperparmameters is done in fit_multi
    #     """
    #     pass

    @abstractmethod
    def fit_multi_task(self, lst_features, lst_targets):
        """
        Method that implements the multitask algorithm that attempts to learn
        some information across datasets. It accepts a
        list of datasets and whatever the appropriate labelling is for the given
        scheme (e.g. trial-specific labels for supervised learning, dataset
        mutlilabels for multilinear transfer learning, etc) and trains an
        internal list of classifier objects. Depending on the algorithm this can
        correspond to a single instance or more. 
        """
        pass

    @abstractmethod
    def fit(self, features, targets):
        """
        Method that implements the transfer learning. Given whatever internal
        representation of the cross-dataset knowledge, and a new dataset and
        associated information, it returns a classifier object.
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

