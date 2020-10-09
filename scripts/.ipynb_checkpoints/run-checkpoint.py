from preprocessing import *
from implementation import *
from helpers import *
from findbestdegree import *
from proj1_helpers import *
from datetime import datetime

OUT_DIR = "../out"



def main():
    print("LOAD DATA")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("FILTERING DATA")
    x_test, x_train = standardize(x_test, x_train)
    x_train, ys_train = remove_outliers(x_train, ys_train)

    print("FINDING DEGREE")
    degrees = np.arange(10,11);
    lambdas = np.logspace(-4, 0, 2)
    best_degree = best_degree_selection(x_train,ys_train,degrees, 2, lambdas);
    print("BUILDING POLYNOMIALS with degree", best_degree)
    tx_train = build_poly(x_train, best_degree)
    tx_test = build_poly(x_test, best_degree)

    print("LEARNING MODEL BY LEAST SQUARES")
    w, mse = least_squares(ys_train, tx_train)

    print("PREDICTING VALUES")
    y_pred = predict_labels(w, tx_test)

    print("EXPORTING CSV")
    create_csv_submission(ids_test, y_pred, "{}/submission-{}.csv".format(OUT_DIR, datetime.now()))


if __name__ == '__main__':
    main()
