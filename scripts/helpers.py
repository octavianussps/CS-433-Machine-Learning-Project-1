# helper functions
import numpy as np

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Gradient descent algorithm.
    
    inputs: 
    	y = labels
    	tx = feature matrix
        initial_w = vector of initial weights
        max_iters = number of maximum iterations on the loop
        gamma :  Step size of the iterative method
    
    output:
    	ws = weights corresponding to logistic regression solution
    	losses = mse loss corresponding to the logistic regression solution
    """
    ws = []
    losses = []
    weight = initial_w
    ws.append(weight)
    for n_iter in range(max_iters):
        # compute loss, gradient
        gradient = compute_gradient(y,tx,weight)
        loss = compute_loss(y,tx,weight)
        if np.isinf(loss):
             raise ValueError("Infinite loss with gamma %.0e, exiting."%gamma)
        # gradient w by descent update
        weight = weight - gamma * gradient
        # store w and loss
        ws.append(weight)
        losses.append(loss)
    return losses, ws

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    builds feature matrix
    
    inputs: 
    	x : features matrix X
        degree : degree used to create the polynomial
    
    output:
    	poly : the features matrix X after polynomial expansion
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def compute_stoch_gradient(y, tx, w):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.
    inputs: 
    	y = labels
    	tx = feature matrix
        w: weight
    
    output:
    	gradient : Gradient for loss function of Mean Squared Error evaluated in w
    """
    N = len(y)
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e) / N
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Stochastic gradient descent algorithm., uses batch_iter algorithm
    inputs: 
    	y = labels
    	tx = feature matrix
        initial_w = vector of initial weights
        max_iters = number of maximum iterations on the loop
        gamma :  Step size of the iterative method
    
    outputs:
    	ws = weights corresponding to stochastic regression solution
    	losses = mse loss corresponding to the stochastic regression solution
    
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            # compute a stochastic gradient and loss
            gradient = compute_stoch_gradient(y_batch,tx_batch,w)
            loss = compute_loss(y,tx,w)
            # update w through the stochastic gradient update
            w = w - gamma * gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>

    inputs: 
    	y = labels
    	tx = feature matrix
        batch_size = data points used included in the batch
        num_batches= Number of batches to be generated
        shuffle=True
    output;
    Iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`

    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]  

def sigmoid(t):
    """
    Apply sigmoid function on t
    input:
        t = Vector in which sigmoid is evaluated
    output:
        Sigmoid function evaluated in t
    """
    return 1.0 / (1 + np.exp(-t))
    
def new_labels(w, tx):
    """
    Generates class predictions given weights, and a test data matrix
    input: 
    	w = weight
    	tx = feature matrix
    output
        y_pred :class predictions given weights, and a test data matrix
    """
    y_prediction = tx.dot(w)
    y_prediction[np.where(y_prediction <= 0.5)] = 0
    y_prediction[np.where(y_prediction > 0.5)] = 1
    return y_prediction

def compute_gradient(y, tx, w, kind='mse'):
    """
    Compute the gradient
    inputs:
        y = labels
    	tx = feature matrix
        w : weight
        kind : mse or mae
    output:
         Gradient for loss function evaluated in w
    raise : NotImplementedError
    """
    error = y - tx.dot(w)
    if kind == 'mse':
        return -tx.T.dot(error)/len(y)
    elif kind == 'mae':
        return -np.sign(error).dot(tx)/len(y) #Sum rows
    else:
        raise NotImplementedError

def calculate_gradient(y, tx, w):
    """
    Compute the gradient of the negative log likelihood loss function
    inputs:
        y = labels
    	tx = feature matrix
        w = weight
    output:
        out = gradient of the negative log likelihood loss function
    """
    y_pred = new_labels(w, tx)
    s = sigmoid(y_pred)
    k = 1.0 / y.shape[0]
    out = k * np.transpose(tx).dot(s - y)
    return out

def compute_loss(y, tx, w, kind='mse'):
    """
    Computes the loss, based on the cost function specified
    inputs:
        y = labels
    	tx = feature matrix
        w : weight
        kind: mae or mse
    output:
        the loss
    raise NotImplementedError
    """
    error = y - tx.dot(w)
    if kind == 'mse':
        return 1/2*np.mean(error**2)
    elif kind == 'mae':
        return sum(np.abs(error))/len(y)
    else:
        raise NotImplementedError


def calculate_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood
    inputs:
        y = labels
    	tx = feature matrix
        w : weight
    output:
        out = loss value by negative log likelihood evaluated in w
    """
    prediction = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(prediction)) + (1 - y).T.dot(np.log(1 - prediction))
    out = np.squeeze(- loss)
    return out

def calculate_loss_reg(y, tx, w, lambda_):
    """
    Compute the cost by negative log likelihood for Regularized Logistic Regression
    inputs:
        y = labels
    	tx = feature matrix
        lambda_: Regularization parameter
    output:
        Loss value by negative log likelihood evaluated in w for Regularized Logistic Regression
    """
    n = tx.shape[0]
    out= calculate_loss(y, tx, w) + (lambda_ / (2 * n)) * np.power(np.linalg.norm(w), 2)
    return out

def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Return the loss and gradient by the algorithm of the penalized logistic regression
    inputs:
        y = labels
    	tx = feature matrix
        w :  weight
        lambda_: Regularization parameter
    output:
        loss
        gradient;
    """
    num_samples = y.shape[0]
    loss = calculate_loss(y, tx, w) + (lambda_ / (2 * num_samples)) * np.power(np.linalg.norm(w), 2)
    gradient = calculate_gradient(y, tx, w) +(1 / tx.shape[0]) * lambda_ * w
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    One step of gradient descent, using the penalized logistic regression.
    inputs:
        y = labels
    	tx = feature matrix
        w =  weight
        gamma =  Step size of the iterative method 
        lambda_ = Regularization parameter
    output:
        w = updated w after 1 step of gradient descent for penalized logistic regression
        loss = after 1 step of gradient descent for penalized logistic regression
        norm of the gradient

    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return w, loss, np.linalg.norm(gradient)

def predict_labels(w, data):
    """
    Generates class predictions given weights, and a test data matrix for Least Squares
    inputs : 
        w: weights
        data: the test data
    output:
        y_prediction : predictions for w and the data matrix for Least Squares
    """
    y_prediction = np.dot(data, w)
    y_prediction[np.where(y_prediction <= 0)] = -1
    y_prediction[np.where(y_prediction > 0)] = 1
    return y_prediction

def predict_labels_log(weights, data):
    """
    Generates class predictions given weights, and a test data matrix for Log
    inputs : 
        w: weights
        data: the test data
    output:
        y_prediction : predictions for w and the data matrix for Least Squares
    """
    y_prediction = np.dot(data, weights)
    y_prediction[np.where(y_prediction <= 0.5)] = -1
    y_prediction[np.where(y_prediction > 0.5)] = 1
    return y_prediction

def calculate_logistic_loss(y, tx, w):
    """Compute the cost by negative log-likelihood."""
    
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)


def calculate_logistic_gradient(y, tx, w):
    """Compute the gradient of loss for sigmoidal prediction."""

    
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)
