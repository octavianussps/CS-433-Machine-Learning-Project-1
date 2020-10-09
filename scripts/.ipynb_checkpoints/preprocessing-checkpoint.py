
def standardize(x_test, x_train):
    """
    Standardizes each column of the train and test data points
    N1 = #train data points
    N2 = #test data points
    D = #number of variables in input data
    :param x_test: Matrix of test data points of size N1xD
    :param x_train: Train data points N2xD
    :return: Standardized matrices x_test and x_train
    """
    for i in range(x_test.shape[1]):
        x_test[:, i], x_train[:, i] = standardize_col(x_test[:, i], x_train[:, i])
    return x_test, x_train

def standardize_col(x1, x2):
    """
    Standardizes arrays x1 and x2 of the train and test data
    :param x1: Array of the train data of size N1x1
    :param x2: Array of the test data of size N2x1
    :return: Standardized arrays x1 and x2
    
    """
    index_x1 = np.where(x1 == -999)
    index_x2 = np.where(x2 == -999)
   
   
    x1_clean = np.delete(x1, index_x1)
    x2_clean = np.delete(x2, index_x2)
    x_clean = np.append(x1_clean, x2_clean)

 
    x1 -= np.mean(x_clean, axis =0)
    x2 -= np.mean(x_clean, axis =0)
    
    x1[index_x1] = 0 
    x2[index_x2] = 0 # where -999
    
    std = np.std(np.append(x1, x2), ddof=1)

    x1 /= std
    x2 /= std
    return x1, x2

def discard_outliers(x_train, ys_train ):
    """
    Discards data points containing outliers as a coordinate (coordinates which have values far from the mean of that column)
    :param x_train: Matrix of input variables of size NxD
    :param ys_train: Vector of labels of size 1xN
    :param threshold: Value indicating the presence of outliers
    :return: Train data without outliers
    """
    index = []
    threshold =8.5
    for i in range(x_train.shape[0]):
        if np.amax(np.abs(x_train[i, :])) > threshold:
            index.append(i)
    x_train = np.delete(x_train, index, 0)
    ys_train = np.delete(ys_train, index, 0)
    return x_train, ys_train