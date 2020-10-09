#!/usr/bin/env python
import numpy as np

######################################
# Helper Functions
######################################

def sigmoid(t):
    """
    Apply sigmoid function on t
    """
    return 1.0 / (1 + np.exp(-t))
    
def new_labels(w, tx):
    """
    Generates class predictions given weights, and a test data matrix
    """
    y_pred = tx.dot(w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def compute_gradient(y, tx, w):
    """
    Compute the gradient
    """
    k = -1.0 / y.shape[0]
    y_pred = tx.dot(w)
    e = y - y_pred
    return k * np.transpose(tx).dot(e)

def calculate_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood loss function
    """
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    k = 1.0 / y.shape[0]
    return k * np.transpose(tx).dot(s - y)

def compute_loss(y, tx, w, kind='mse'):
    """
    MSE or MAE loss functions.
    """
    error = y - tx.dot(w)
    if kind == 'mse':
        return error.dot(error)/(2*len(y))
    elif kind == 'mae':
        return sum(np.abs(error))/len(y)
    else:
        raise NotImplementedError


def calculate_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    """
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_loss_reg(y, tx, w, lambda_):
    """
    Compute the cost by negative log likelihood for Regularized Logistic Regression
    """
    n = tx.shape[0]
    out= calculate_loss(y, tx, w) + (lambda_ / (2 * n)) * np.power(np.linalg.norm(w), 2)
    return out

def penalized_logistic_regression(y, tx, w, lambda_):
    """
    return the loss and gradient.
    """
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    One step of gradient descent, using the penalized logistic regression.
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return w, loss, np.linalg.norm(gradient)


def least_squares(y, tx):
    """calculates the least squares solution 
    using the normal equation to compute the weights,
    solves X.T*X*w = X.T * y (gradient of the loss function, w is the unknown) 
    
    input: 
    	y = labels
    	tx = feature matrix
    
    output:
    	weights = weights corresponding to the least squares solution
    	mse = mse loss corresponding to the least squares solution
    
    """	
    weights = np.linalg.solve(tx.T.dot(tx),tx.T.dot(y))
    mse = compute_mse(y,tx,weights) 
    return weights, mse

    
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """calculates the least squares solution using 
    gradient descent to compute the weights.
    
    input: 
    	y = labels
    	tx = feature matrix
    	initial_w = initial value for the weights
    	max_iters = number of steps
    	gamme = learning rate
    
    output:
    	weights = weights corresponding to the last step
    	mse = mse loss corresponding to the last step
    
    """
    
    losses, ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
    weights = ws[-1]
    mse = losses[-1] 
    return weights, mse
    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """calculates the least squares solution using stochastic 
    gradient descent to compute the weights. Uses batch_iter from
    proj1_helpers.py to compute a minibatch for the dataset.
    
    input: 
    	y = labels
    	tx = feature matrix
    	initial_w = initial value for the weights
    	max_iters = number of steps
    	gamme = learning rate
    
    output:
    	weights = weights corresponding to the last step
    	mse = mse loss corresponding to the last step
  
    """
    
    # default batch_size = 1
    batch_size = 1
    losses, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
    weights = ws[-1]
    mse = losses[-1] 
    return weights, mse


################################################
# Ridge Regression
################################################

def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression
    """
    txt = np.transpose(tx)
    w = np.linalg.solve((txt.dot(tx) + lambda_ * 2 * y.shape[0] * np.identity(tx.shape[1])), txt.dot(y))
    return w, compute_loss(y, tx, w)


################################################
# Logistic Regression
################################################

# Threshold condition (can me modified)
THRESHOLD = 1e-6

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
   # initializing the weight
    losses = []
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w

    # logistic regression
    for i in range(max_iters):
        # learning by gradient descent
        grad = calculate_gradient(y, tx, w)
        loss = calculate_loss(y, tx, w)
        w -= gamma * grad
        losses.append(loss)
        # stop
        if np.abs(losses[-1] - losses[-2]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
        if i > 100 and losses[-1] > losses[-100]:
            gamma = gamma / 10
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    applies regularized logistic regression using stochastic gradient descent to optimize w
    """
    # initializing the weight
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    losses = []

     # regularized logistic regression
    for i in range(max_iters):
        w, loss, grad_norm = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        # stop
        if i > 100 and np.abs(losses[-1] - losses[-100]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
        if i > 100 and losses[-1] > losses[-100]:
            gamma = gamma / 10
    return w, losses[-1]
