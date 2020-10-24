from preprocessing import *
from implementations import *
from helpers import *
from findbestdegree import *
from proj1_helpers import *
from datetime import datetime


OUT_DIR = "../out"



def main():
    print("loading data")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("preprocessing data")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = remove_outliers(x_train, ys_train)
    
    print("finding best degree")
    degrees = np.arange(1,15)
    lambdas = np.logspace(-4, 0, 10)
    lambdas = np.concatenate((0,lambdas),axis=None)
    lambda_ = 100
    
    # Cross-Validation 
    k_fold = 4
    # best degree selection uses cross validation to find the best lambda/degree
    #best_degree = find_best_degree(x_train, ys_train, degrees, k_fold, lambdas);
    best_degree = 10
    print("building polynomial with degree", best_degree)
    tx_train = build_poly(x_train, best_degree)
    tx_test = build_poly(x_test, best_degree)


    print("training model with least squares")
    wInit, mse = least_squares(ys_train, tx_train)
	
    max_iters = 100
    gamma = 0.1**(20)
    w,mse = least_squares_GD(ys_train, tx_train, None, max_iters, gamma)

    #w,mse = least_squares_SGD(ys_train, tx_train, w, max_iters, gamma)
    #w,mse = logistic_regression(ys_train, tx_train, wInit, max_iters, gamma)
    #w, mse = reg_logistic_regression(ys_train, tx_train, lambda_, None, max_iters, gamma)
    #w, mse = ridge_regression(ys_train, tx_train, lambda_)
    #w,mse = least_squares_GD(ys_train, tx_train, initial_w, max_iters, gamma)
	
    print("predicting labels for test data")
    y_pred = predict_labels(w, tx_test)
    # y_pred = predict_labels_log(w, data)

    print("exporting csv file")
    name_out = "{}/submission.csv".format(OUT_DIR)
    create_csv_submission(ids_test, y_pred, "{}/submission1.csv".format(OUT_DIR))
    #create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
