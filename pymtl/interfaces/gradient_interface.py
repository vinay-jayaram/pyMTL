#!/usr/bin/env python

from abc import ABCMeta, abstractmethod

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"


class GradientInterface(object):
    """
    Abstract class interface to retrieve the gradient for feature decomposition
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_loss_gradient(self, features, targets):
        pass
