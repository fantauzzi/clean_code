# Predict Customer Churn

### Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The project has a didactic purpose, it illustrates the very basics of loading
a dataset, performing some EDA, features engineering, evaluation and 
interpretation of the results; it has the essentials of the hygiene for software 
development in place, inclusive of automated unit testing.

The dataset is the "Credit Card customers" from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Files and data description
```
.
├── churn_library.py -> Library functions for the Jupyter notebook.
├── churn_notebook.ipynb -> A Jupyter notebook illustrating a basic supervised learning workflow 
├── churn_script_logging_and_tests.py -> Automated unit-tests for churn_library.py
├── data -> Directory with the dataset
├── encoded_set.pickle -> Intermediate computation file used by unit-tests, regenerated as needed
├── images -> Dir. with images for EDA and with model test results and intepretation 
├── logs -> Dir. with log from unit-tests
├── models -> Dir. where models are saved in a binary format after training, ready for inference
├── README.md -> This README file
├── requirements_py3.8.txt -> list of packages 
└── train_test_set.pickle -> Intermediate computation file used by unit-tests, regenerated as needed
```

## Running Files

The project requires Python 3.8. It has been tested on Ubuntu 20.04.

Clone the git repository

`git clone git@github.com:fantauzzi/clean_code.git`

cd into the repo

`cd clean_code`

Make sure you have the necessary Python packages installed

`pip install -r requirements_py3.8.txt`

To run the tests

`python churn_script_logging_and_tests.py`

If there are no errors, the script will terminate after outputting the models
training and test results to console, like

```
-> python churn_script_logging_and_tests.py
random forest results
test results
              precision    recall  f1-score   support

           0       0.96      0.99      0.98      2543
           1       0.93      0.80      0.86       496

    accuracy                           0.96      3039
   macro avg       0.95      0.90      0.92      3039
weighted avg       0.96      0.96      0.96      3039

train results
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      5957
           1       1.00      1.00      1.00      1131

    accuracy                           1.00      7088
   macro avg       1.00      1.00      1.00      7088
weighted avg       1.00      1.00      1.00      7088

logistic regression results
test results
              precision    recall  f1-score   support

           0       0.90      0.97      0.93      2543
           1       0.72      0.44      0.54       496

    accuracy                           0.88      3039
   macro avg       0.81      0.70      0.74      3039
weighted avg       0.87      0.88      0.87      3039

train results
              precision    recall  f1-score   support
```

It may take several minutes, depending on the hardware. It is possible to check
the test progress in the log file:

`tail -f logs/churn_library.log`

The content of log file for a successful unit-test run looks like this:

```
-> cat logs/churn_library.log
root - INFO - Testing import_data()
root - INFO - Testing import_data: SUCCESS
root - INFO - Testing perform_eda()
root - INFO - Executing perform_eda: SUCCESS
root - INFO - Testing encoder_helper()
root - INFO - Testing encoder_helper: SUCCESS
root - INFO - Testing perform_feature_engineering()
root - INFO - Testing perform_feature_engineering: SUCCESS
root - INFO - Testing train_models()
root - INFO - Testing train_models: SUCCESS
```

You can open and run the Jupyter Notebook `churn_notebook.ipynb` from
[JupyterLab](https://jupyter.org/). The notebook loads the data, perform its EDA, 
features engineering and then trains the model. Note that it may take several minutes to complete,
depending on the hardware.