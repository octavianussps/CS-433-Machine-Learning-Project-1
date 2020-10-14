# helper functions
import numpy as np

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    print("using gradient descent{}".format(len(y)))
    # Define parameters to store w and loss
    ws = []
    
    losses = []
    w = initial_w
    ws.append(w)
    for n_iter in range(max_iters):
       
        gradient = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        if np.isinf(loss):
             raise ValueError("Infinite loss in least_squares_GD with gamma %.0e, exiting."%gamma)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
         #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    N = len(y)
    e = y - tx.dot(w)
    gradient = -tx.T.dot(e) / N
    return gradient
    #raise NotImplementedError


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
            gradient = compute_stoch_gradient(y_batch,tx_batch,w)
            loss = compute_loss(y,tx,w)
            w = w - gamma * gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
        #print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
           #   bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
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

def compute_gradient(y, tx, w, kind='mse'):
    """
    Compute the gradient
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

def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix for Least Squares
    :param weights: Vector of weights of size 1x(1+(DG*D))
    :param data: Matrix containing the test data
    :return: Class predictions for given weights and a test data matrix for Least Squares
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred