#!/usr/bin/env python
import numpy as np
from helpers import *
######################################
# least squares Functions
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
    #weights = np.linalg.lstsq(tx,y)
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
    w = initial_w
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
    Implement ridge regression
    """
    a = tx.T.dot(tx)+2*tx.shape[0]*lambda_*np.eye(tx.shape[1])
    b = tx.T.dot(y)
    wRidge = np.linalg.solve(a,b)
    return wRidge, compute_loss(y, tx, wRidge)
    


################################################
# Logistic Regression
################################################

THRESHOLD = 1e-8

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    """
    losses = []
    # initializing the weight in the case there is none
    if (initial_w is None):
        initial_w = np.zeros(tx.shape[1])
    w = initial_w

    # logistic regression
    for i in range(max_iters):
        # learning by gradient descent
        grad = calculate_logistic_gradient(y, tx, w)
        # compute loss by negative log likelihood
        loss = calculate_logistic_loss(y, tx, w)
        #loss /= tx.shape[1]
        print("loss",loss)
        w -= gamma * grad
        losses.append(loss)
        # convergence criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < THRESHOLD:
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

    # regularized logistic regression, choose big max_iters because of really small lambda
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
