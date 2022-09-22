"""
This file contains various functions to support EDA, features engineering, model training and results visualization
and interpretation for the Credit Card customers dataset. Produced images are saved into the `images` directory.

author: Udacity and https://github.com/fantauzzi

creation date: 21-Sep-2022
"""
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# import libraries
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('images/eda/churn_hist.png')

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('images/eda/customer_age_hist.png')

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_bar.png')

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('images/eda/total_trans_ct_hist.png')

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/heatmap.png')


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category, new_category_name in zip(category_lst, response):
        encoded_values = []
        encoded_groups = data_frame.groupby(category).mean()['Churn']

        for val in data_frame[category]:
            encoded_values.append(encoded_groups.loc[val])

        data_frame[new_category_name] = encoded_values

    return data_frame


def perform_feature_engineering(data_frame, response=None):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y = data_frame['Churn']
    X = pd.DataFrame()

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    if not response:
        response = [f'{category}_Churn' for category in cat_columns]
    data_frame = encoder_helper(data_frame, category_lst=cat_columns, response=response)

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = data_frame[keep_cols]
    # X.head()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    is_interactive = plt.isinteractive()
    if is_interactive:
        plt.ioff()
    plt.figure(figsize=(5, 6))
    plt.rc('figure', figsize=(5, 6))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/random_forest_res.png')

    plt.figure(figsize=(5, 6))
    plt.rc('figure', figsize=(5, 6))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('images/results/logistic_regr_res.png')
    if is_interactive:
        plt.ion()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test, show_plot=True):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                   X_test, y_test, ax=axes, alpha=0.8)
    lrc_plot.plot(ax=axes, alpha=0.8)
    if show_plot:
        plt.show()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)

    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(rfc_model, X_test, y_test, ax=axes, alpha=0.8)
    lrc_plot.plot(ax=axes, alpha=0.8)
    if show_plot:
        plt.show()

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=show_plot)

    feature_importance_plot(cv_rfc.best_estimator_,
                            X_test, r'./images/results/feature_importance.png')

    classification_report_image(y_train=y_train,
                                y_test=y_test,
                                y_train_preds_lr=y_train_preds_lr,
                                y_train_preds_rf=y_train_preds_rf,
                                y_test_preds_lr=y_test_preds_lr,
                                y_test_preds_rf=y_test_preds_rf
                                )
