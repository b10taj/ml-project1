"""Required ML methods for Project 1 (NumPy only)."""

import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        (w, loss): last weight vector and its MSE loss (with factor 0.5)
    """
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of SGD steps
        gamma: step size

    Returns:
        (w, loss): last weight vector and its MSE loss (with factor 0.5)
    """
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)

    Returns:
        (w, loss): optimal weights and its MSE loss (with factor 0.5)
    """
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        lambda_: regularization parameter

    Returns:
        (w, loss): optimal weights and its MSE loss (without penalty term)
    """
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        (w, loss): last weight vector and its logistic loss
    """
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        lambda_: regularization parameter (penalty lambda * ||w||^2)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        (w, loss): last weight vector and its logistic loss (without penalty)
    """
    raise NotImplementedError
