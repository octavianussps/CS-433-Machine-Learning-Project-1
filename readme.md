
# Machine Learning Project 1
**Marion Chabrier, Valentin Margraf, Octavianus Sinaga**

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report. 


### **Project structure**


| Folder | Files |
| ------ | ----------- |
| data | test and training data in .csv format |
| out    | contains the final subission file, also csv.format |
| latex-example-paper    | contains the pdf report of our project |
| scripts   | all the python scripts we used in this project, further explanation below.
 + *implementations.py* contains the six regression methods.
 + *helpers.py* contains several helper functions to run the regression methods such as compute_loss or compute_gradient.
 + *findbestdegree.py* is a script which uses cross validation in order to find out the best degree and/or the best lambda for the different methods as well as functions to load and submit the data.
+ *plots.py* visualizes the affect of different choices of hyperparameters on the RMSE of the method respectively.
+ *preprocessing.py* preprocesses the data i.e. standardizes it and removes outliers and missing values.
+ *run.py* is a script which produces the submission.csv file for the test data. 

### **Prediction of the Higgs Boson**

In order to submit the predictions we give on the test data, you have to run the *run.py* file. It will load the data, preprocess it,
build the feature matrix and then train the model (in this case Least Squares). Then the model is used to predict the labels of the test data and the submission.csv file will be generated.

It will give some output like this:

```
loading data
preprocessing data
building polynomial with degree 11
training model with least squares
predicting labels for test data
exporting csv file
```
