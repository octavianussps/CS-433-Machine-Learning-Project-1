from preprocessing import *
from helpers import *
from implementations import *
from proj1_helpers import *

OUT_DIR = "../out"
DEGREE = 10
lambda_=100

def main():
    
    print("loading data")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("filtering data")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = remove_outliers(x_train, ys_train)

    print("building polynomial with degree 10")
    tx_train = build_poly(x_train, DEGREE)

    print("learning model by least Square")
    w_ls, mse_ls = least_squares(ys_train, tx_train)
    
    w_ini = None
    
    print("learning model by gradient descent")
    max_iters = 100
    gamma = 0.01**(20)
    w_gd, mse_gd = least_squares_GD(ys_train, tx_train, w_ini, max_iters, gamma)

    print("learning model by stochastic gradient descent")
    max_iters = 100
    gamma = 0.01**(20)
    w_sgd, mse_sgd = least_squares_SGD(ys_train, tx_train, w_ini, max_iters, gamma)
    
    print("learning model by ridge regression")
    max_iters = 100
    gamma = 0.01**(20)
    w_rr, mse_rr = ridge_regression(ys_train, tx_train, lambda_)
    
    print("learning model by logistic regression")
    max_iters = 100
    gamma = 0.01**(20)
    w_lr, mse_lr = logistic_regression(ys_train, tx_train, w_ini, max_iters, gamma)
    
    print("learning model by reg logistic regression")
    max_iters = 100
    gamma = 0.01**(20)
    w_rlr, mse_rlr =reg_logistic_regression(ys_train, tx_train, lambda_, w_ini, max_iters, gamma)

    print("LEAST SQUARES\t \t \t \tW: {} \tMSE:{}".format(w_ls[-1], mse_ls))
    print("GRADIENT DESCENT\t \t \tW: {} \tMSE:{}".format(w_gd[-1], mse_gd))
    print("RIDGE REGRESSION\t \t \tW: {} \tMSE:{}".format(w_rr[-1], mse_rr))
    print("STOCHASTIC GRADIENT DESCENT\t \tW: {} \tMSE:{}".format(w_sgd[-1], mse_sgd))
    print("STOCHASTIC LOGISTIC REGRESSION\t \tW: {} \tMSE:{}".format(w_lr[-1], mse_lr))
    print("STOCHASTIC REG LOGISTIC REGRESSION\tW: {} \tMSE:{}".format(w_rlr[-1], mse_rlr))

if __name__ == '__main__':
    main()