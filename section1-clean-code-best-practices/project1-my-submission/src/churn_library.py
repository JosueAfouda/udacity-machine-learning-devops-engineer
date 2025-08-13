"""
This is the churn_library.py python file that we used to find customers who are likely to churn
The execution of this file will produce artefacts in images and models folders.
Date: March 26, 2023
Author: Josue AFOUDA
"""
# import libraries
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    list_columns = [
        col for col in df.columns if col not in [
            'Unnamed: 0', 'CLIENTNUM']]

    df_corr = df[list_columns].corr()

    list_columns.append('Heatmap')

    for column_name in list_columns:
        plt.figure(figsize=(16, 14))
        if column_name == 'Heatmap':
            sns.heatmap(
                df_corr,
                mask=np.triu(np.ones_like(df_corr, dtype=bool)),
                center=0, cmap='RdBu', linewidths=1, annot=True,
                fmt=".2f", vmin=-1, vmax=1
            )
        else:
            if df[column_name].dtype != 'O':
                df[column_name].hist()
            else:
                sns.countplot(data=df, x=column_name)
        plt.savefig("images/eda/" + column_name + ".jpg")
        plt.close()


def encoder_helper(df, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument]

    output:
            df: pandas dataframe with new columns for
    '''

    for column_name in category_lst:
        column_lst = []
        column_groups = df.groupby(column_name)[response].mean()

        for val in df[column_name]:
            column_lst.append(column_groups.loc[val])

        df[column_name + "_" + response] = column_lst

    return df


def perform_feature_engineering(df, response="Churn"):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    # Features
    X = df[keep_cols]

    # Target
    y = df[response]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3, random_state=42
    )

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

    class_reports_dico = {
        "Logistic Regression train results": classification_report(y_train, y_train_preds_lr),
        "Logistic Regression test results": classification_report(y_test, y_test_preds_lr),
        "Random Forest train results": classification_report(y_train, y_train_preds_rf),
        "Random Forest test results": classification_report(y_test, y_test_preds_rf)
    }

    for title, report in class_reports_dico.items():
        plt.rc('figure', figsize=(7, 3))
        plt.text(
            0.2, 0.3, str(report), {
                'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.title(title, fontweight='bold')
        plt.savefig("images/results/" + title + ".jpg")
        plt.close()


def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values

    output:
             None
    '''

    vars_imp = pd.Series(
        model.feature_importances_, index=X_data.columns
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 12))
    sns.barplot(x=vars_imp, y=vars_imp.index)
    plt.xlabel("Importance")
    plt.title("Features Importances")
    plt.savefig("images/results/features_importances.jpg")
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
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

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # Predictions with Random Forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Predictions with Logistic Regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # ROC curves images
    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig("images/results/roc_curves.jpg")
    plt.close()

    # Clasification reports images
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Fetures importances images
    feature_importance_plot(cv_rfc.best_estimator_, X_test)

    # Model Serialization
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    logging.info("Importing data")
    data = import_data("./data/bank_data.csv")

    logging.info("Performing EDA")
    perform_eda(data)

    logging.info("Encoding categorical features")
    data_encoded = encoder_helper(
        data,
        category_lst=['Gender',
                      'Education_Level',
                      'Marital_Status',
                      'Income_Category',
                      'Card_Category']
    )

    logging.info("Performing Feature Engineering")
    Xtrain, Xtest, ytrain, ytest = perform_feature_engineering(data_encoded)

    logging.info("Training ML algorithms and Store artefacts (images + models)")
    train_models(Xtrain, Xtest, ytrain, ytest)