
  

# Machine Learning Project 1

## **Authors**
  * Marion Chabrier - marion.chabrier@epfl.ch
  * Valentin Margraf - valentin.margraf@epfl.ch
  * Octavianus Sinaga - octavianus.sinaga@epfl.ch

This readme file contains useful information of the structure of the project containing the code and report. For further information of the project and its results we advise you to read the report.

The detailed explanation of the project is on the report (`latexreport/report.pdf`).

  
  

## **Project structure**



| Folder  | Files |
|:--:|:--:|
| `data/`  | test and training data in .csv format |
| `out/`  | contains the final subission file, also csv.format |
| `latexreport/` | contains the pdf and the latex report of our project |
| `scripts/`  | all the python scripts we used in this project, further explanation below |

In `scripts/` we can see:
+  `implementations.py` contains the main implementations of the ML algorithms required in the project statement.

+  `helpers.py` contains several helper functions to run the regression methods such as compute_loss or compute_gradient.

+  `proj1_helpers.py` is used to load the data, predict labels for the test data and to create a submission.

+  `findbestdegree.py` is a script which uses cross validation in order to find out the best degree and/or the best lambda for the different methods as well as functions to load and submit the data.

+ `checkAllImplementation.py` is a script with run all programm in implementation and give us the MSE and the last weight obtained

+ `checkPreprocessing.ipynb` is a script with help us to understand the preprocessing

+  `plots.py` visualizes the affect of different choices of hyperparameters on the RMSE of the method respectively.

+  `preprocessing.py` preprocesses the data i.e. standardizes it and removes outliers and missing values.

+  `run.py` is a script which produces the submission.csv file for the test data.

  

## **Prediction of the Higgs Boson**


The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles
have mass.  We will apply machine learning techniques to recreate the process of discovering the Higgs particle.
If you're interested in more background on this dataset, we point you to the longer description here: *https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf*.

## **Prerequisites**
`Python 3+` 
Assuming python is installed, install `numpy` (if not already installed) with one this command:
        
   + `pip install numpy`
  + `conda install numpy`
## **Dataset**
You will find the testing and training data in the folder named `Data`.
Please open this file and unzip both `test.csv.zip` and `train.csv.zip` to extract the csv files.
A `Sample-submission.csv` file is also there to show how your output should look like when submitting to Kaggle.

## **Running**
In order to submit the predictions we give on the test data, you have to run the `run.py` file. It will load the data, preprocess it, build the feature matrix and then train the model (in this case Least Squares). Then the model is used to predict the labels of the test data and the `out/submission.csv` file will be generated.

  

It will give some output like this:


```
loading data

preprocessing data

building polynomial with degree 11

training model with least squares

predicting labels for test data

exporting csv file
```

## **Results**

We achieved a categorical accuracy of 0.822 and a F1- score of  0.728 on the website. All the details are in the report.
