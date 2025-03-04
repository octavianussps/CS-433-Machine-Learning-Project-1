from preprocessing import *
from implementations import *
from helpers import *
from findbestdegree import *
from proj1_helpers import *
from datetime import datetime
import matplotlib.pyplot as plt



def cross_validation_visualization_degrees(degrees, mse_tr, mse_te):
    """
    visualization the curves of mse_tr and mse_te.
    inputs :
        degree: List of degrees to be tested 
        mse_tr : List of train error values for the different tested degrees
        mse_te : List of test error values for the different tested degrees
    output:
        plot with the given values
    """
    plt.plot(degrees, mse_tr, marker=".", color='b', label='train error')
    plt.plot(degrees, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("degree")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")

def cross_validation_visualization_lamda(lamda, mse_tr, mse_te):
    """
    visualization the curves of mse_tr and mse_te.
    inputs :
        lamda: List of lamdas to be tested 
        mse_tr : List of train error values for the different tested degrees
        mse_te : List of test error values for the different tested degrees
    output:
        plot with the given values
    """
    plt.plot(degrees, mse_tr, marker=".", color='b', label='train error')
    plt.plot(degrees, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lamdas")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def cross_validation_demo():
     """
    Demo of the cross validation
    output:
        plot with the given values
    """
    #initialization
    seed = 12
    degrees = np.arange(2,7)
    degree = 12
    k_fold = 4
    k_indices = build_k_indices(y, k_fold, seed)
    lambda_ = 0.1
    rmse_tr = []
    rmse_te = []
    # cross validation
    for degree in degrees:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te,_ = cross_validation(y, x, k_indices, k, lambda_, degree)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    cross_validation_visualization_degrees(degrees, rmse_tr, rmse_te)



if __name__ == '__main__':
    
    print("loading data")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("preprocessing data (standazation and remove outliers")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = remove_outliers(x_train, ys_train)
    
    y = ys_train
    x = x_train
    cross_validation_demo()
    
