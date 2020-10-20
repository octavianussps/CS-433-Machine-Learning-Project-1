# Machine Learning Project 1
Marion Chabrier, Valentin Margraf, Octavianus Sinaga

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report. 


### **Project structure**


| Folder | Files |
| ------ | ----------- |
| data | test and training data in .csv format |
| out    | contains the final subission file, also csv.format |
| latex-example-paper    | contains the pdf report of our project |
| scripts   | all the python scripts we used in this project.
 + implementations.py contains the six regression methods.
 + helpers.py contains several helper functions to run the regression methods such as compute_loss or compute_gradient.
 + findbestdegree.py is a script which uses cross validation in order to find out the best degree and/or the best lambda for the different methods.
+ plots.py visualizes the affect of different choices of hyperparameters on the RMSE of the method respectively.
+ preprocessing.py preprocesses the data i.e. standardizes it and removes outliers and missing values.
+ run.py is a script which produces the submission.csv file for the test data. 
ipt which produces the submission.csv files for the test data. since least squares gave the best result
