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
    degrees = np.arange(10,12);
    #lambdas = np.logspace(-4, 0, 20)
    # Cross-Validation 
    #k_fold = 4
    # best degree selection uses ridge_regression to find the best lambda
    #best_degree = best_degree_selection(x_train, ys_train, degrees, k_fold, lambdas);
    
    #print("building polynomial with degree", best_degree)
    best_degree=10
    tx_train = build_poly(x_train, best_degree)
    tx_test = build_poly(x_test, best_degree)

    print("training model with least squares")
    w, mse = least_squares(ys_train, tx_train)
	
    #print("LEARNING MODEL BY logRegression")
    #max_iters = 30
    #gamma = 0.1
    #lambda_ = 0.1
    #w, mse = reg_logistic_regression(ys_train, tx_train, lambda_, None, max_iters, gamma)
	
    print("predicting labels for test data")
    y_pred = predict_labels(w, tx_test)

    print("exporting csv file")
    name_out = "{}/submission.csv".format(OUT_DIR)
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
