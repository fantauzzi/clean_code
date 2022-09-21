import os
import errno
import logging
import pandas as pd
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models
from pathlib import Path
from imageio.v2 import imread
import pickle

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

train_test_set_pickle = 'train_test_set.pickle'
encoded_pickle = 'encoded_set.pickle'


def test_import(import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    expected_files = ['churn_hist.png', 'customer_age_hist.png',
                      'heatmap.png', 'marital_status_bar.png', 'total_trans_ct_hist.png']
    eda_path = r'./images/eda/'
    expected_files = [eda_path + file_name for file_name in expected_files]
    # If the files perform_eda should create exist already, rename them appending .bck to them
    for file_name in expected_files:
        if Path(file_name).exists():
            new_name = file_name + '.bck'
            Path(file_name).rename(Path(new_name))
    try:
        df = import_data("./data/bank_data.csv")
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    except Exception as ex:
        logging.error(
            f"Testing test_eda: cannot continue with test, got this exception while importing data {ex}"),
        raise ex
    try:
        perform_eda(df)
        logging.info("Executing perform_eda: SUCCESS")
    except Exception as ex:
        logging.error(f"Executing import_eda: got exception {ex}")
        raise ex
    for file_name in expected_files:
        if not Path(file_name).exists():
            logging.error(
                f"Testing test_eda: file {file_name} was not produced")
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), file_name)
        try:
            imread(file_name)
        except Exception as ex:
            logging.error(
                f"Testing test_eda: while trying to load image {file_name} got this exception {ex}")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = import_data(r"./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    response = [f'{category}_Churn' for category in cat_columns]
    try:
        res = encoder_helper(df, category_lst=cat_columns, response=response)
    except Exception as err:
        logging.error("Testing encode_helper: error while performing th encoding.")
        raise err

    with open(encoded_pickle, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        assert type(df) is pd.DataFrame
    except AssertionError as err:
        logging.error("Testing encode_helper: returned value should be of type pandas.DataFrame")
        raise err
    try:
        assert len(res) == len(df)
    except AssertionError as err:
        logging.error("Testing encode_helper: returned dataframe should have the same length as the input dataframe")
        raise err
    try:
        cat_encoded_columns = set(res.columns)
        for col in cat_columns:
            assert col + '_Churn' in cat_encoded_columns
    except AssertionError as err:
        logging.error(
            "Testing encode_helper: categorical column names should now end in '_Churn'")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # Load pickled dataframe, that will be used for testing
    try:
        with open(encoded_pickle, 'rb') as f:
            df = pickle.load(f)
    except Exception as err:
        logging.error(
            f"Testing perform_feature_engineering: cannot load pickle file {encoded_pickle} for testing, test cannot proceed")
        raise err
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: error while performing feature engineering.")
        raise err
    # Pickle the result of feature engineering, it will be the input to another test
    pickl_me = (X_train, X_test, y_train, y_test)
    with open(train_test_set_pickle, 'wb') as f:
        pickle.dump(pickl_me, f, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(df)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: number of samples in resulting dataset splits is inconsistent.")
        raise err
    try:
        assert (X_train.columns == X_test.columns).all()
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                     'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                     'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                     'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                     'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                     'Income_Category_Churn', 'Card_Category_Churn']
        assert len(X_train.columns) == len(keep_cols)
        columns_set = set(X_train.columns)
        for col in keep_cols:
            assert col in columns_set
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: train and test datasets must have the same variables, and only the expected variables")
        raise err
    logging.info("Testing perform_feature_engineering: SUCCESS")


def test_train_models(train_models):
    '''
    test train_models
    '''
    files_to_be_generated = ['models/logistic_model.pkl',
                             'models/rfc_model.pkl',
                             'images/results/random_forest_res.png',
                             'images/results/logistic_regr_res.png',
                             'images/results/feature_importance.png']

    # Load pickled file with dataset already split between training and test
    try:
        with open(train_test_set_pickle, 'rb') as f:
            split_dataset = pickle.load(f)
    except Exception as ex:
        logging.error(
            f"Testing train_models: loading of file {train_test_set_pickle} with pickled dataset failed, test can't proceed.")
        raise ex

    # Remove files that should be generated during training, if they exist
    for filename in files_to_be_generated:
        Path(filename).unlink(missing_ok=True)

    # Train the model
    try:
        train_models(*split_dataset, show_plot=False)
    except Exception as ex:
        logging.error(
            f"Testing train_models: error while training model.")
        raise ex

    # Check that files that should have been generated during training are actually there
    for filename in files_to_be_generated:
        try:
            assert Path(filename).exists()
        except AssertionError as err:
            logging.error(
                f"Testing train_models: file {filename} was not generated during training.")
            raise err

    logging.info("Testing train_models: SUCCESS")


if __name__ == "__main__":
    logging.info('Testing import_data()')
    test_import(import_data)
    logging.info('Testing perform_eda()')
    test_eda(perform_eda)
    logging.info('Testing encoder_helper()')
    test_encoder_helper(encoder_helper)
    logging.info('Testing perform_feature_engineering()')
    test_perform_feature_engineering(perform_feature_engineering)
    logging.info('Testing train_models()')
    test_train_models(train_models)
