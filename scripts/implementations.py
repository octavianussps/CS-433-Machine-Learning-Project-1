#!/usr/bin/env python
import numpy as np
from helpers import *


######################################
# Least Squares Functions
######################################



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
    mse = compute_loss(y,tx,weights) 
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
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w.astype(float)
    losses, ws = gradient_descent(y, tx, w, max_iters, gamma)
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
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w
    # default batch_size = 1
    batch_size = 1
    losses, ws = stochastic_gradient_descent(y, tx, w, batch_size, max_iters, gamma)
    weights = ws[-1]
    mse = losses[-1] 
    return weights, mse


################################################
# Ridge Regression
################################################

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    input: 
    	y = labels
    	tx = feature matrix
        lambda_ :  Regularization parameter
    
    output:
    	w = weights corresponding to ridge regression solution
    	mse = mse loss corresponding to the ridge regression solution
    
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
    
    input: 
    	y = labels
    	tx = feature matrix
        initial_w = vector of initial weights
        max_iters = number of maximum iterations on the loop
        gamma :  Step size of the iterative method
    
    output:
    	w = weights corresponding to logistic regression solution
    	losses[-1] = mse loss corresponding to the logistic regression solution
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
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < THRESHOLD:
            gamma = gamma / 10
            if gamma < 1e-10:
                break
        if i > 100 and losses[-1] > losses[-100]:
            gamma = gamma / 10
    return w, losses[-1]


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    
    Regularized Logistic regression using gradient descent
    
    input: 
    	y = labels
    	tx = feature matrix
        initial_w = vector of initial weights
        max_iters = number of maximum iterations on the loop
        gamma :  Step size of the iterative method
    
    output:
    	w = weights corresponding to logistic regression solution
    	losses[-1] = mse loss corresponding to the logistic regression solution
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
