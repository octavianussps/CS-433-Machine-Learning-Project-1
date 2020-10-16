import numpy as np




def standardize(x_test, x_train):
    """
    standardizes the train and test data matrices
    
    input: 
        x_test: matrix which contains test data
        x_train: matrix which contains train data
    return:
        standardized matrices x_test, x_train
    """
    for i in range(x_test.shape[1]):
        x_test[:, i], x_train[:, i] = standardize_col(x_test[:, i], x_train[:, i])
    
    return x_test, x_train

def standardize_col(x1, x2):
    """
    standardizes arrays of train and test data 
    after having set -999 values to 0
    
    input:
        x_1: column of (test) data matrix
        x_2: column of (train) data matrix
    return:
        standardized columns x_1,x_2
    """
    index_x1 = np.where(x1 == -999)
    index_x2 = np.where(x2 == -999)
   
   
    x1_clean = np.delete(x1, index_x1)
    x2_clean = np.delete(x2, index_x2)
    x_clean = np.append(x1_clean, x2_clean)

 
    x1 = x1 - np.mean(x_clean, axis =0)
    x2 = x2 - np.mean(x_clean, axis =0)
    x1[index_x1] = 0 
    x2[index_x2] = 0 # where -999
    
    std = np.std(np.append(x1, x2), ddof=1)

    x1 = x1/std
    x2 = x2/std
    return x1, x2

def remove_outliers(x_train, ys_train):
    """
    discards data points containing outliers, 
    i.e. values being far away from the mean 
    
    input: 
        x_train: matrix which contains train data
        ys_train: array which contains labels
    return: 
        train and label data without outliers
    """
    index = []
    threshold = 8.5
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > threshold:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)
    return x_train, ys_train
