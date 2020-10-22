import numpy as np
from helpers import *
from implementations import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k +  1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, x, k_indices, k, lambda_, degree):
    # 4-fold, hence k = 4, k-th group is for testings, the other k-1
    # groups are for training
    subgroups = k_indices
    testIndices = subgroups[k]
    trainIndices = np.delete(subgroups,k,axis=0)
    trainIndices = np.concatenate((trainIndices[0],trainIndices[1],trainIndices[2]),axis=None)
    xTrain = x[trainIndices]
    xTest = x[testIndices]
    yTrain = y[trainIndices]
    yTest = y[testIndices]
    
    #build data
    featureMatrixTest = build_poly(xTest,degree)
    featureMatrixTrain = build_poly(xTrain,degree)

    #wTrain,_ = ridge_regression(yTrain,featureMatrixTrain,lambda_)
    #wTrain,_ = logistic_regression(yTraiyTrainn, featureMatrixTrain, None, 40, 0.01)
    #wTrain,_ = least_squares_SGD(yTrain, featureMatrixTrain, None, 30, 0.01)
    #wTrain,_ = ridge_regression(yTrain, featureMatrixTrain, lambda_)
    wTrain,_ = least_squares(yTrain, featureMatrixTrain)
    #wTrain,_ = reg_logistic_regression(yTrain, featureMatrixTrain, lambda_, None, 30, 0.1**(17))
       
    # calculate MSE for train and test data !! using the training weights !!
    loss_tr = np.sqrt(2*compute_loss(yTrain, featureMatrixTrain, wTrain))
    loss_te = np.sqrt(2*compute_loss(yTest, featureMatrixTest, wTrain))
    
    return loss_tr, loss_te, wTrain


def find_best_degree(x,y,degrees, k_fold, lambdas, seed = 1):
    '''find out the best degree of the model 
    in order to minimize the resulting loss    
    '''
    
    k_indices = build_k_indices(y, k_fold, seed)
    
    # store in lists (rmse and lambdas)
    best_lambdas = []
    best_rmses = []
    # iterate over given array of degrees
    for degree in degrees:
        #store test errors
        rmse_te = []
        for lambda_ in lambdas:
            rmse_te_tmp = []
            # perform cross validation
            for k in range(k_fold):
                _, loss_te, _ = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te_tmp.append(loss_te)
            rmse_te.append(np.mean(rmse_te_tmp))
        #argmin gives smallest rmse
        ind_lambda_opt = np.argmin(rmse_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        #print("degree",degree,"lambda",lambdas[ind_lambda_opt])
        best_rmses.append(rmse_te[ind_lambda_opt])
        
    ind_best_degree =  np.argmin(best_rmses)      
        
    return degrees[ind_best_degree]

