"""Required ML methods for Project 1 (NumPy only)."""

import numpy as np


def sigmoid(t):
    """Logistic function 1 / (1 + exp(-t)), applied elementwise.

    Args:
        t: input values, any shape

    Returns:
        sigmoid of t, same shape as t
    """
    return 1 / (1 + np.exp(-t))


def mse_loss(y, tx, w):
    """MSE loss with the 0.5 factor: 0.5 * mean((y - tx w)^2).

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        scalar MSE loss
    """
    return 0.5 * np.mean((y - tx.dot(w)) ** 2)


def mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -tx^T (y - tx w) / N.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        gradient w.r.t. w, shape (D,)
    """
    err = y - tx.dot(w)
    return -tx.T.dot(err) / len(err)


def logistic_loss(y, tx, w):
    """Logistic loss: mean(log(1 + exp(tx w)) - y * tx w).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        scalar logistic loss
    """
    # logaddexp(0, z) = log(1 + exp(z)) without overflow for large z.
    return np.mean(np.logaddexp(0, tx.dot(w)) - y * tx.dot(w))


def logistic_gradient(y, tx, w):
    """Gradient of the logistic loss: tx^T (sigmoid(tx w) - y) / N.

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        gradient w.r.t. w, shape (D,)
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


def gradient_descent(grad_fn, initial_w, max_iters, gamma):
    """Run gradient steps w <- w - gamma * grad_fn(w).

    Args:
        grad_fn: function mapping w, shape (D,), to a gradient, shape (D,)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        last weight vector, shape (D,)
    """
    w = initial_w
    for i in range(max_iters):
        w = w - gamma * grad_fn(w)
    return w


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
    w = gradient_descent(lambda w: mse_gradient(y, tx, w), initial_w, max_iters, gamma)
    return w, mse_loss(y, tx, w)


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

    def stochastic_grad(w):
        idx = np.random.randint(len(y))
        return mse_gradient(y[idx : idx + 1], tx[idx : idx + 1], w)

    w = gradient_descent(stochastic_grad, initial_w, max_iters, gamma)
    return w, mse_loss(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)

    Returns:
        (w, loss): optimal weights and its MSE loss (with factor 0.5)
    """

    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, mse_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        lambda_: regularization parameter

    Returns:
        (w, loss): optimal weights and its MSE loss (without penalty term)
    """

    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    return w, mse_loss(y, tx, w)


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

    w = gradient_descent(
        lambda w: logistic_gradient(y, tx, w), initial_w, max_iters, gamma
    )
    return w, logistic_loss(y, tx, w)


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

    w = gradient_descent(
        lambda w: logistic_gradient(y, tx, w) + 2 * lambda_ * w,
        initial_w,
        max_iters,
        gamma,
    )
    return w, logistic_loss(y, tx, w)
