from preprocessing import *
from helpers import *
from implementation import *
from proj1_helpers import *

OUT_DIR = "../out"
DEGREE = 11


def main():
    
    print("LOAD DATA")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = remove_outliers(x_train, ys_train)

    print("BUILDING POLYNOMIALS")
    tx_train = build_poly(x_train, DEGREE)

    print("LEARNING MODEL BY LEAST SQUARES")
    w_ls, mse_ls = least_squares(ys_train, tx_train)
    print("MSE LS \n MSE{}".format(mse_ls))

    w_ini = w_ls
    
    print("LEARNING MODEL BY GRADIENT DESCENT")
    max_iters = 500
    gamma = 0.01
    w_gd, mse_gd = least_squares_GD(ys_train, tx_train, w_ini, max_iters, gamma)
    print("MSE GD \n  MSE{}".format(mse_gd))

    print("LEARNING MODEL BY STOCHASTIC GRADIENT DESCENT")
    max_iters = 5000
    gamma = 0.01
    batch_size = 10000
    w_sgd, mse_sgd = least_squares_SGD(ys_train, tx_train, w_ini, batch_size, max_iters, gamma)
    print("MSE SGD: \n  MSE{}".format(mse_sgd))



if __name__ == '__main__':
    main()