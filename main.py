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
    conn = sqlite3.connect('database.sqlite')
    data = pd.read_sql("""SELECT Match.id, 
                            League.name AS league_name, 
                            season, 
                            stage, 
                            shoton,
                            shotoff,
                            goal,
                            corner,
                            foulcommit,
                            card,
                            TAH.team_api_id AS home_team_api_id,
                            TAA.team_api_id AS away_team_api_id,
                            home_team_goal, 
                            away_team_goal,
                            TAH.buildUpPlaySpeedClass AS home_buildUpPlaySpeedClass,
                            TAH.buildUpPlayDribblingClass AS home_buildUpPlayDribblingClass,
                            TAH.buildUpPlayPassingClass AS home_buildUpPlayPassingClass,
                            TAH.buildUpPlayPositioningClass AS home_buildUpPlayPositioingClass,
                            TAH.chanceCreationPassingClass AS home_chanceCreationPassingClass,
                            TAH.chanceCreationCrossingClass AS home_chanceCreationCrossingClass,
                            TAH.chanceCreationShootingClass AS home_chanceCreationShootingClass,
                            TAH.chanceCreationPositioningClass AS home_chanceCreationPositioingClass,
                            TAH.defencePressureClass AS home_defencePressureClass,
                            TAH.defenceAggressionClass AS home_defenceAggressionClass,
                            TAH.defenceTeamWidthClass AS home_defenceTeamWidthClass,
                            TAH.defenceDefenderLineClass AS home_defenceDefenderLineClass,

                            TAA.buildUpPlaySpeedClass AS away_buildUpPlaySpeedClass,
                            TAA.buildUpPlayDribblingClass AS away_buildUpPlayDribblingClass,
                            TAA.buildUpPlayPassingClass AS away_buildUpPlayPassingClass,
                            TAA.buildUpPlayPositioningClass AS away_buildUpPlayPositioingClass,
                            TAA.chanceCreationPassingClass AS away_chanceCreationPassingClass,
                            TAA.chanceCreationCrossingClass AS away_chanceCreationCrossingClass,
                            TAA.chanceCreationShootingClass AS away_chanceCreationShootingClass,
                            TAA.chanceCreationPositioningClass AS away_chanceCreationPositioingClass,
                            TAA.defencePressureClass AS away_defencePressureClass,
                            TAA.defenceAggressionClass AS away_defenceAggressionClass,
                            TAA.defenceTeamWidthClass AS away_defenceTeamWidthClass,
                            TAA.defenceDefenderLineClass AS away_defenceDefenderLineClass
                    FROM Match
                    JOIN League on League.id = Match.league_id
                    LEFT JOIN Team_Attributes AS TAH on TAH.team_api_id = Match.home_team_api_id
                    LEFT JOIN Team_Attributes AS TAA on TAA.team_api_id = Match.away_team_api_id
                    WHERE season not like '2015/2016' and goal is not null
                    LIMIT 700000;""", conn)
    return data


# This method clean the data from none values
def pre_processing(data):
    threshold = int(0.7 * len(data))
    # This loop remove all features that the number of non none values is at least 70%.
    for col in data.columns:
        if threshold > data[col].count():
            data.drop(col, axis='columns', inplace=True)
    # Clean all row that have more then 5 features with None values.
    data.dropna(thresh=5, inplace=True)
    return data


# This method creates the goal variable of the data
def create_goal_var(data):
    return data


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


def fill_nones(train, test):
    print(train)


def evaluate_model(model, data, X, y):
    k_fold = KFold(n_splits=10)
    accuracy_list = []
    for train_index, test_index in k_fold.split(data):
        train_predictors = X.iloc[train_index, :]
        train_target = y.iloc[train_index]
        test_predictors = X.iloc[test_index, :]
        test_target = y.iloc[test_index]
        fill_nones(train_predictors, test_predictors)
        model.fit(train_predictors, train_target)
        model_prediction = model.predict(test_predictors)
        accuracy = accuracy_score(test_target, model_prediction)
        accuracy_list.append(accuracy)

    print("Cross-Validation Accuracy Score: %s" % "{0:.3}".format(mean(accuracy_list)))
    return model


if __name__ == '__main__':
    # read data from squlite
    data = get_data()
    # clean data
    data = pre_processing(data)
    # create goal variable (y)
    data = create_goal_var(data)
    # Have the Data as X, y

