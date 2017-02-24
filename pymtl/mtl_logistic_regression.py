#!/usr/bin/env python

import scipy
import warnings
import numpy as np
from pymtl.mtl_linear_regression import BayesRegressionClassifier
from pymtl.interfaces.mtl_priors import GaussianParams, SKGaussianParams
from pymtl.interfaces.gradient_interface import GradientInterface

__author__ = "Karl-Heinz Fiebig"
__copyright__ = "Copyright 2017"

def sigmoid(s):
    """
    Convenience for sigmoid function
    """
    return 1.0/(1+np.exp(-s))

class BayesLogisticClassifier(BayesRegressionClassifier, GradientInterface):
    """
    Class that implements a logostic classifier with Gaussian prior regularization
    """
    
    def __init__(self, max_prior_iter=1000, prior_conv_tol=1e-4, lam=1, lam_style='ML', optim_algo='gd', pred_threshold=0.5):
        """
        TODO
        """
        super(BayesLogisticClassifier, self).__init__(max_prior_iter, prior_conv_tol, lam, lam_style)
        self.optim_algo = optim_algo
        self.pred_threshold = pred_threshold
        self._classes = None
        self._prior = None
        self._weights = None
        self._invSigma = None
        self._set_internal_classes([0,1])


    def fit(self, features, targets):
        """
        Class-specific implementation of the fit function given features, targets, and prior
        """
        # data safety
        if features.shape[0] != targets.shape[0]:
            raise ValueError('Number of samples in data set ({}) does not match number of \
                             samples ({}) in the target vector'.format(features.shape[0],
                                                                       targets.shape[0]))
        X_train = features
        y_train, self._classes = self._convert_classes(targets)
        # Setup prior if not already done
        if self._prior is None:
            self.init_model(X_train.shape, y_train.shape)
        max_iter = 10000
        conv_tol = 0.01
        verbose='warn'
        if self.optim_algo == 'gd':
            self._minimize_crossentropy_gd(X_train, y_train, max_iter, conv_tol, verbose)
        if self.optim_algo == 'agd':
            self._minimize_crossentropy_agd(X_train, y_train, max_iter, conv_tol, verbose)
        if self.optim_algo == 'cg':
            self._minimize_crossentropy_cg(X_train, y_train, verbose)
        return self


    def predict(self, features):
        """
        Given new data points, return the most likely class
        """

        # Return probabilities above prediction threshold
        pred = self.predict_proba(features) > self.pred_threshold
        return self._recover_classes(pred)

    def predict_proba(self, features):
        """
        Given data points, return the probability of class 1
        """
        # Check arguments
        assert features.shape[1] == len(self._prior.mu), \
            'feature dimensionality is not compatible with this model!'
        if self._weights is None:
            w = self._prior.mu
        else:
            w = self._weights
        prob = sigmoid(features.dot(w))
        return prob

    def predict_log_proba(self, features):
        """
        Convenience function for log probability (deprecated?)
        """
        return np.log(self.predict_proba(features))

    def loss(self, features, targets):
        """
        TODO
        """
        X = features
        y = self._convert_classes(targets)[0]
        
        if self._weights is None:
            w = self._prior.mu
        else:
            w = self._weights
        err = self._cross_entropy_error(X, y, w=w)
        return err

    @BayesRegressionClassifier.prior.setter
    def prior(self, prior):
        """
        TODO
        """
        self._attr_prior = prior
        self._invSigma = np.linalg.inv(self._prior.Sigma)

    ###########################################################################
    # Auxiliary private methods used internally by this model
    ###########################################################################

    def get_loss_gradient(self, features, targets):
        """Gradient of the Cross-Entropy FD loss w.r.t. this models weights.

        Parameters
        ----------
        X : matrix, shape = (n_samples, k, d)
        y : vector, shape = (n_samples, 1)

        Returns
        -------
        gradient : vector, shape = (d,)
            The Cross-Entropy loss gradient w.r.t. the spectral weights.
        """
        X = features
        y, self._classes = self._convert_classes(targets)
        if self._weights is None:
            w = self._prior.mu
        else:
            w = self._weights
        grad = self._crossentropy_grad(X, y, w)
        return grad.reshape((len(grad), 1))

    def _minimize_crossentropy(self, X, y, max_iter=10000, tol=0.01, verbose='warn'):
        """Minimizes the Cross-Entropy Loss objective w.r.t. the weights
        using the optimizer specified by self.optim_algo.

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
        if self.optim_algo == 'gd':
            self._minimize_crossentropy_gd(
                X, y, max_iter, tol, verbose)
        if self.optim_algo == 'agd':
            self._minimize_crossentropy_agd(
                X, y, max_iter, tol, verbose)
        if self.optim_algo == 'cg':
            self._minimize_crossentropy_cg(
                X, y, verbose)


    def _minimize_crossentropy_cg(self, X, y, verbose):
        """Minimizes the Cross-Entropy Loss objective w.r.t. the weights.

        This method performs a Newton-CG optimization to minimize the
        Cross-Entropy loss objective by iteratively evaluating different weight
        estimates. In the process, repeated calls to the loss function and
        gradient are used, which compose the bottle-neck for performance of
        this minimization technique.

        Parameters
        ----------
        X : matrix, shape = (n_samples, n_features)
            The training set from which to minimize the loss.

        y : vector, shape = (n_samples, 1)
            True labels in {1, 0} for each training point in the corresponding
            training set.

        verbose : bool, default True
            If set  True, additional information on the minimization progress
            is printed to the console.

        Returns
        -------
        w : vector, shape = (n_features, 1)
            The optimal weight vector resulting from the minimization process
            on the samples X and y.
        """
        # Retrieve useful variables
        n_samples = len(X)
        d = len(self._prior.mu)
        # Check arguments
        assert X.shape[1] == d, \
            'feature dimensionality is not compatible with this model!'
        assert y.shape[0] == X.shape[0], \
            'number of samples do not agree!'
        assert y.shape[1] == 1, \
            ('Labels have to be a column vector, but is of shape ')
        # Create initial weight vector if none was set before
        w0 = np.copy(self._prior.mu)
        # Minimize multi-task cross entropy loss
        if verbose == 'enable':
            print('Init loss:' + str(self._cross_entropy_error(X, y, w0)))
        f_loss = lambda param: self._cross_entropy_error(X, y, param.flatten())
        f_grad = lambda param: self._crossentropy_grad(X, y, param.flatten())
        # f_hess = lambda param: self._crossentropy_hessian(X, y, param.flatten())
        f_hess = None
        result = scipy.optimize.minimize(fun=f_loss, x0=w0, method='Newton-CG',
                                         jac=f_grad, hess=f_hess)
        self._weights = result.x.reshape((d, 1))
        if verbose and not result.success:
            print('Warning! Optimization convergence failed!')
            print(result.message)
        # Return final vector and evaluation
        if verbose == 'enable':
            print('Final loss: ' + str(self._cross_entropy_error(X, y, self._weights)))


    def _minimize_crossentropy_agd(self, X, y, max_iter, tol, verbose):
        """Minimizes the Cross-Entropy Loss objective w.r.t. the weights
        using gradient descent with adaptive parameter-wise learning rates.

        Learning rates are adapted with additive increase and multiplicative
        decrease on detection of sign switches.

        Parameters
        ----------
        X : matrix, shape = (n_samples, d)
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
        d = X.shape[1]
        inc_rate = 0.01
        dec_rate = 0.75
        v_lvl = 100
        # Initialize weights and learning rates
        self._weights = np.zeros(self._mu_w.shape)
        eta_w = np.array([0.0001] * d).reshape((d, 1))
        eta_global = 1.0
        grad_w = np.zeros((d, 1))
        # Start gradient descent optimization
        if verbose == 'enable':
            print('Initial CE error:' + str(self._cross_entropy_error(X, y, w=self._weights)))
        for i in range(max_iter):
            w_prev = self._weights
            grad_w_prev = grad_w
            # Update spectral weights
            grad_w = self._crossentropy_grad(X, y, self._weights).reshape((d, 1))
            self._weights = self._weights - eta_global * eta_w * grad_w
            # Check convergence status
            conv_w = np.sum(np.abs(self._weights - w_prev) < tol * np.abs(w_prev))
            if conv_w == d:
                break
            # Find all sign switches
            s_w = grad_w_prev * grad_w < 0
            ns_w = np.logical_not(s_w)
            # Cease learning rates on overshooting rates
            eta_w[s_w] = dec_rate * eta_w[s_w]
            # Increase learning rates on rest
            eta_w[ns_w] = eta_w[ns_w] + inc_rate * eta_w[ns_w]
        # Print final information and return optimal weights
        if verbose == 'enable':
            print('Final CE error: ' + str(self._cross_entropy_error(X, y, w=self._weights)))
        if verbose and i == max_iter - 1:
            warnings.warn('FD-AGD minimization did not converge after ' +
                          str(max_iter) + ' iterations!')


    def _minimize_crossentropy_gd(self, X, y, max_iter, tol, verbose):
        """Minimizes the Cross-Entropy Loss objective w.r.t. the weights
        using gradient descent with adaptive global learning rate.

        The learning rate is adapted with additive increase and multiplicative
        decrease on detection of error increase (i.e. overshooting).

        Parameters
        ----------
        X : matrix, shape = (n_samples, d)
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
        d = X.shape[1]
        inc_rate = 0.1
        dec_rate = 0.025
        v_lvl = 100
        # Initialize weights and learning rates
        eta = 0.1
        self._weights = np.copy(self._prior.mu)
        ce_current = self._cross_entropy_error(X, y, w=self._weights)
        # Start gradient descent optimization
        if verbose == 'enable':
            print('Initial CE error: ' + str(ce_current))
        for i in range(max_iter):
            ce_prev = ce_current
            w_prev = self._weights
            # Update spectral weights
            grad_w = self._crossentropy_grad(X, y, self._weights).reshape((d, 1))
            self._weights = self._weights - eta * grad_w
            # Check convergence status
            ce_current = self._cross_entropy_error(X, y, w=self._weights)
            diff_w = np.abs(self._weights - w_prev)
            conv_w = np.sum(diff_w < tol * np.abs(w_prev))
            diff_ce = np.abs(ce_prev - ce_current)
            if diff_ce < tol * 1e-3:
                break
            # Adapt learning rate according to current error development
            if ce_current >= ce_prev:
                # Decrease learning rate and withdraw iteration
                eta = dec_rate * eta
                self._weights = w_prev
                ce_current = ce_prev
            else:
                # Slowly increase learning rate
                eta = eta + inc_rate * eta


    def _crossentropy_grad(self, X, y, w):
        """Returns the gradient of the Cross-Entropy Loss w.r.t. the weights.

        This method computes the partial derivative of the Cross-Entropy
        loss function w.r.t. the weight vector and can be used in gradient
        based optimization techniques to estimate the optimal weights regarding
        the loss.

        Parameters
        ----------
        X : matrix, shape = (n_samples, n_features)
            The training set from which to learn a model.

        y : vector, shape = (n_samples, 1)
            True labels in {1, 0} for each training point in the corresponding
            training set.

        w : vector, shape = (n_features, 1) or (n_features,), default None
            Optional explicit parameter vector to use for the gradient
            computation. If this argument is not provided (or set to None),
            the weight vector self._w of this model is used to compute the
            gradient.

        Returns
        -------
        gradient : vector, shape = (n_features,)
            The Cross-Entropy loss gradient w.r.t. the weights.
        """
        # Retrieve useful variables
        n_samples = len(X)
        n_features = len(self._prior.mu)
        # Check arguments
        assert X.shape[1] == n_features, \
            'feature dimensionality is not compatible with this model!'
        assert y.shape[0] == X.shape[0], \
            'number of samples do not agree!'
        assert y.shape[1] == 1, \
            ('Labels have to be a column vector, but is of shape ' + y.shape)
        # Reshape flatten weight vector if necessary
        if len(w.shape) == 1:
            w = w.reshape((len(w), 1))
        # Compute gradient
        diff = sigmoid(X.dot(w)) - y
        penalty_w = (w - self._prior.mu)
        # Compute final gradient and add penalty term
        grad = X.T.dot(diff)
        grad = self._prior.Sigma.dot(grad) + self.lam*penalty_w
        return grad.flatten()


    def _crossentropy_hessian(self, X, y, w):
        """Returns the Hessian of the Cross-Entropy Loss w.r.t. the weights.

        This method computes the second partial derivative of the Cross-Entropy
        loss function w.r.t. the weight vector and can be used in optimization
        techniques that can make use of the objecitve's curvature information
        to estimate the optimal weights.

        Parameters
        ----------
        X : matrix, shape = (n_samples, n_features)
            The training set from which to learn a model.

        y : vector, shape = (n_samples, 1)
            True labels in {1, 0} for each training point in the corresponding
            training set.

        w : vector, shape = (n_features, 1) or (n_features,), default None
            Optional explicit parameter vector to use for the gradient
            computation. If this argument is not provided (or set to None),
            the weight vector self._w of this model is used to compute the
            gradient.

        Returns
        -------
        gradient : vector, shape = (n_features,)
            The Cross-Entropy loss gradient w.r.t. the weights.
        """
        # Retrieve useful variables
        n_samples = len(X)
        n_features = len(self._prior.mu)
        # Check arguments
        assert X.shape[1] == n_features, \
            'feature dimensionality is not compatible with this model!'
        assert y.shape[0] == X.shape[0], \
            'number of samples do not agree!'
        assert y.shape[1] == 1, \
            ('Labels have to be a column vector, but is of shape ' + y.shape)
        # Reshape flatten weight vector if necessary
        if len(w.shape) == 1:
            w = w.reshape((len(w), 1))
        # Compute Hessian
        h = sigmoid(X.dot(w))
        penalty = self._invSigma
        XhX = X.T.dot(np.diag((h * (1 - h)).flatten()).dot(X))
        return XhX + self.lam * penalty

    # def _cross_entropy_error(self, X, y, w=None):
    #     """Computes the Cross-Entropy Loss in a training set.
    #
    #     This method computes the Cross-Entropy error function with prior
    #     information from the given labeled data set and the current prior
    #     in this model.
    #
    #     Important: this function is mainly called for optimization of the
    #     weight vector, therefore constant terms that do not depend on the
    #     weights are droped in order to avoid expensive overhead!
    #
    #     Parameters
    #     ----------
    #     X : matrix, shape = (n_samples, n_features)
    #         The training set from which to learn a model.
    #
    #     y : vector, shape = (n_samples, 1)
    #         True labels in {1, 0} for each training point in the corresponding
    #         training set.
    #
    #     w : vector, shape = (n_features, 1) or (n_features,), default None
    #         Optional explicit parameter vector to use for the loss computation.
    #         If this argument is not provided (or set to None), the weight
    #         vector self._w of this model is used to compute the loss.
    #
    #     Returns
    #     -------
    #     loss : float
    #         The Cross-Entropy error (omitting terms that do not depend on the
    #         weights).
    #     """
    #     # Retrieve useful variables
    #     n_samples = len(X)
    #     n_features = len(self._prior.mu)
    #     # Check arguments
    #     assert X.shape[1] == n_features, \
    #         'feature dimensionality is not compatible with this model!'
    #     assert y.shape[0] == X.shape[0], \
    #         'number of samples do not agree!'
    #     assert y.shape[1] == 1, \
    #         ('Labels have to be a column vector, but is of shape')
    #     # Use model weights if not explicitly given
    #     if w is None:
    #         w = self._w
    #     # Reshape flatten weight vector if necessary
    #     if len(w.shape) == 1:
    #         w = w.reshape((len(w), 1))
    #     # Compute individual terms of the cross entropy loss
    #     # Note: In order to avoid numerical difficulties when the log arguments
    #     # are close to zero, a small value of 1e-6 is added to those arguments.
    #     h = sigmoid(X.dot(w))
    #     ce = -np.sum(y*np.log(h+1e-9) + (1-y)*np.log(1-h+1e-9))
    #     zm = (w - self._prior.mu)
    #     penalty = 0.5*zm.T.dot(self._invSigma.dot(zm)).flatten()[0]
    #     return ce + self.lam*penalty

    def _cross_entropy_error(self, X, y, w):
        # Compute individual terms of the cross entropy loss
        # Note: In order to avoid numerical difficulties when the log arguments
        # are close to zero, a small value of 1e-6 is added to those arguments.
        h = sigmoid(X.dot(w))
        ce = -np.sum(y*np.log(h+1e-100) + (1-y)*np.log(1-h+1e-100))
        zm = (w - self._prior.mu)
        penalty = 0.5*zm.T.dot(self._invSigma.dot(zm)).flatten()[0]
        return ce + self.lam*penalty
