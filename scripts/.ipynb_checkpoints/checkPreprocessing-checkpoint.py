from preprocessing import *
from helpers import *
from implementations import *
from proj1_helpers import *
import matplotlib.pyplot as plt

OUT_DIR = "../out"
DEGREE = 11


def main():
    
    print("LOAD DATA")
    ys_train, x_train, ids_train = load_csv_data("../data/train.csv")
    y_test, x_test, ids_test = load_csv_data("../data/test.csv")

    print("FILTERING DATA")
    print("check standardize")
    x_test, x_train = standardize(x_test, x_train)
    plt.boxplot(X_train)
    print("check removal")
    x_train, ys_train = remove_outliers(x_train, ys_train)


if __name__ == '__main__':
    main()