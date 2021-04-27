import sys
import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from numpy.core import mean
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold



# get data from sqlite
def get_data():
    try:
        conn = sqlite3.connect("database.sqlite")
    except sqlite3.Error as e:
        print(e)
    return conn


# This method adds missing categorials values
def add_categorial_missing_values(data, col):
    data[col].fillna(data[col].mode()[0], inplace=True)


# This method adds missing numerical values
def add_numerical_missing_values(data, col):
    data[col].fillna(data[col].mean(), inplace=True)


# This method choose best params model by using grid search algorithm
def best_params_model(model, X, y, grid_variables):
    k_fold = KFold(n_splits=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    grid = GridSearchCV(model, grid_variables, scoring='f1', cv=k_fold, refit=True)
    grid_results = grid.fit(X_train, y_train)
    model = grid_results.best_estimator_
    return model


def fill_nones(train_predictors, train_target, test_predictors, test_target):
    pass


def evaluate_model(model, data, X, y):
    k_fold = KFold(n_splits=10)
    accuracy_list = []
    for train_index, test_index in k_fold.split(data):
        train_predictors = X.iloc[train_index, :]
        train_target = y.iloc[train_index]
        test_predictors = X.iloc[test_index, :]
        test_target = y.iloc[test_index]
        fill_nones(train_predictors, train_target, test_predictors, test_target)
        model.fit(train_predictors, train_target)
        model_prediction = model.predict(test_predictors)
        accuracy = accuracy_score(test_target, model_prediction)
        accuracy_list.append(accuracy)

    print("Cross-Validation Accuracy Score: %s" % "{0:.3}".format(mean(accuracy_list)))
    return model


if __name__ == '__main__':
    # read data from squlite
    conn = get_data()

    # Have the Data as X, y
