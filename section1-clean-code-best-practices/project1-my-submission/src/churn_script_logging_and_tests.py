"""
This file contains unit tests for the churn_library.py library
The execution of this file will produce .log files in logs folder
Date: March 27, 2023
Author: Josue AFOUDA
"""
import os
import logging
import pytest
import joblib
from churn_library import import_data, perform_eda, encoder_helper, \
    perform_feature_engineering, train_models


logging.basicConfig(
    filename='./logs/churn_script_logging_and_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(path)
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
    return df


def test_eda(df):
    '''
    test perform eda function
    '''
    perform_eda(df)

    files_names = os.listdir('./images/eda')

    for file_name in files_names:
        try:
            with open('./images/eda/' + file_name, 'r', encoding="utf-8"):
                logging.info("Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error("Testing perform eda: image files do not exist")
            raise err


def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    encoded_columns = [
        'Gender_Churn',
        'Education_Level_Churn'
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]

    df_encoded = encoder_helper(
        df,
        category_lst=['Gender',
                      'Education_Level',
                      'Marital_Status',
                      'Income_Category',
                      'Card_Category']
    )

    try:
        set(encoded_columns).issubset(df_encoded.columns.to_list())
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: fails to generate encoded columns")
        raise err
    return df_encoded


def test_perform_feature_engineering(df_encoded):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df_encoded)
    try:
        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("The division of data was not done well")
        raise err
    return X_train, X_test, y_train, y_test


def test_train_models(X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    results_images = os.listdir('./images/results')

    for file_name in results_images:
        try:
            with open('./images/results/' + file_name, 'r', encoding="utf-8"):
                logging.info(
                    "Testing train_models (Generation of Reports): SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                "Testing train_models (Generation of Reports): image files do not exist")
            raise err

    try:
        joblib.load("models/rfc_model.pkl")
        joblib.load("models/logistic_model.pkl")
        logging.info("Testing train_models (Model Serialization): SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models (Generation of Reports): image files do not exist")
        raise err


if __name__ == "__main__":
    print("test_import_data")
    raw_data = test_import(path="./data/bank_data.csv")

    print("test_eda")
    test_eda(raw_data)

    print("test_encoder_helper")
    encoded_data = test_encoder_helper(raw_data)

    print("test_perform_feature_engineering")
    Xtrain, Xtest, ytrain, ytest = test_perform_feature_engineering(
        encoded_data)

    print("test_train_models")
    test_train_models(Xtrain, Xtest, ytrain, ytest)
