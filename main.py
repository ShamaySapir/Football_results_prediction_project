import json
import xmltodict
import sys
# from bs4 import BeautifulSoup
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

categorial_vec = ['h_playSpeed', 'a_playSpeed', 'h_playDribbling', 'a_playDribbling', 'h_playPassing',
                  'a_playPassing', 'h_playPositioing', 'a_playPositioing', 'h_creationPassing',
                  'a_creationPassing', 'h_creationCrossing', 'a_creationCrossing', 'h_creationShooting',
                  'a_creationShooting', 'h_creationPositioing', 'a_creationPositioing', 'h_defencePressure',
                  'a_defencePressure', 'h_defenceAggression', 'a_defenceAggression', 'h_defenceTeamWidth',
                  'a_defenceTeamWidth', 'h_defenceDefenderLine', 'a_defenceDefenderLine']


# get train data from sqlite
def get_train_data():
    conn = sqlite3.connect('database.sqlite')
    data = pd.read_sql("""SELECT Match.id AS match_id,
                            League.name AS league_name, 
                            season, 
                            stage, 
                            shoton,
                            shotoff,
                            goal,
                            corner,
                            foulcommit,
                            card,
                            TAH.team_api_id AS home_team_id,
                            TAA.team_api_id AS away_team_id,
                            home_team_goal, 
                            away_team_goal,
                            TAH.buildUpPlaySpeedClass AS h_playSpeed,
                            TAH.buildUpPlayDribblingClass AS h_playDribbling,
                            TAH.buildUpPlayPassingClass AS h_playPassing,
                            TAH.buildUpPlayPositioningClass AS h_playPositioing,
                            TAH.chanceCreationPassingClass AS h_creationPassing,
                            TAH.chanceCreationCrossingClass AS h_creationCrossing,
                            TAH.chanceCreationShootingClass AS h_creationShooting,
                            TAH.chanceCreationPositioningClass AS h_creationPositioing,
                            TAH.defencePressureClass AS h_defencePressure,
                            TAH.defenceAggressionClass AS h_defenceAggression,
                            TAH.defenceTeamWidthClass AS h_defenceTeamWidth,
                            TAH.defenceDefenderLineClass AS h_defenceDefenderLine,

                            TAA.buildUpPlaySpeedClass AS a_playSpeed,
                            TAA.buildUpPlayDribblingClass AS a_playDribbling,
                            TAA.buildUpPlayPassingClass AS a_playPassing,
                            TAA.buildUpPlayPositioningClass AS a_playPositioing,
                            TAA.chanceCreationPassingClass AS a_creationPassing,
                            TAA.chanceCreationCrossingClass AS a_creationCrossing,
                            TAA.chanceCreationShootingClass AS a_creationShooting,
                            TAA.chanceCreationPositioningClass AS a_creationPositioing,
                            TAA.defencePressureClass AS 'a_defencePressure',
                            TAA.defenceAggressionClass AS a_defenceAggression,
                            TAA.defenceTeamWidthClass AS a_defenceTeamWidth,
                            TAA.defenceDefenderLineClass AS a_defenceDefenderLine
                    FROM Match
                    JOIN League on League.id = Match.league_id
                    LEFT JOIN Team_Attributes AS TAH on TAH.team_api_id = Match.home_team_api_id
                    LEFT JOIN Team_Attributes AS TAA on TAA.team_api_id = Match.away_team_api_id
                    WHERE season not like '2015/2016' and goal is not null
                    LIMIT 50000;""", conn)
    print("Got train data succssefully")
    return data


# get test data from sqlite
def get_test_data():
    conn = sqlite3.connect('database.sqlite')
    data = pd.read_sql("""SELECT Match.id AS match_id,
                            League.name AS league_name, 
                            season, 
                            stage, 
                            shoton,
                            shotoff,
                            goal,
                            corner,
                            foulcommit,
                            card,
                            TAH.team_api_id AS home_team_id,
                            TAA.team_api_id AS away_team_id,
                            home_team_goal, 
                            away_team_goal,
                            TAH.buildUpPlaySpeedClass AS h_playSpeed,
                            TAH.buildUpPlayDribblingClass AS h_playDribbling,
                            TAH.buildUpPlayPassingClass AS h_playPassing,
                            TAH.buildUpPlayPositioningClass AS h_playPositioing,
                            TAH.chanceCreationPassingClass AS h_creationPassing,
                            TAH.chanceCreationCrossingClass AS h_creationCrossing,
                            TAH.chanceCreationShootingClass AS h_creationShooting,
                            TAH.chanceCreationPositioningClass AS h_creationPositioing,
                            TAH.defencePressureClass AS h_defencePressure,
                            TAH.defenceAggressionClass AS h_defenceAggression,
                            TAH.defenceTeamWidthClass AS h_defenceTeamWidth,
                            TAH.defenceDefenderLineClass AS h_defenceDefenderLine,

                            TAA.buildUpPlaySpeedClass AS a_playSpeed,
                            TAA.buildUpPlayDribblingClass AS a_playDribbling,
                            TAA.buildUpPlayPassingClass AS a_playPassing,
                            TAA.buildUpPlayPositioningClass AS a_playPositioing,
                            TAA.chanceCreationPassingClass AS a_creationPassing,
                            TAA.chanceCreationCrossingClass AS a_creationCrossing,
                            TAA.chanceCreationShootingClass AS a_creationShooting,
                            TAA.chanceCreationPositioningClass AS a_creationPositioing,
                            TAA.defencePressureClass AS 'a_defencePressure',
                            TAA.defenceAggressionClass AS a_defenceAggression,
                            TAA.defenceTeamWidthClass AS a_defenceTeamWidth,
                            TAA.defenceDefenderLineClass AS a_defenceDefenderLine
                    FROM Match
                    JOIN League on League.id = Match.league_id
                    LEFT JOIN Team_Attributes AS TAH on TAH.team_api_id = Match.home_team_api_id
                    LEFT JOIN Team_Attributes AS TAA on TAA.team_api_id = Match.away_team_api_id
                    WHERE season like '2015/2016' and goal is not null
                    LIMIT 50000;""", conn)
    print("Got test data succssefully")
    return data


def convert_xml_to_json_feature(xml):
    json_xml = np.NaN
    if xml is not None:
        dict_xml = xmltodict.parse(xml)
        json_xml = json.dumps(dict_xml)
    return json_xml


def xml_change_values(data):
    features = ['shoton', 'shotoff', 'goal', 'corner', 'foulcommit', 'card']
    for feature in features:
        data[feature] = data[feature].apply(lambda x: convert_xml_to_json_feature(x))
    print("xml data was succssefully converted")
    return data

# def convert_xml_to_json_feature(xml):
#     if xml is None:
#         return json.dumps({})
#
#     json_xml = xmltodict.parse(xml)
#     json_feature = json.dumps(json_xml)
#     return json_feature
#
#
# def xml_change_values(data):
#     features = ['shoton', 'shotoff', 'goal', 'corner', 'foulcommit', 'card']
#     for feature in features:
#         data[[feature]] = data[[feature]].apply(lambda x: convert_xml_to_json_feature(x[feature]), axis=1,
#                                             result_type='broadcast')
#     return data


# This method drop unnecessary features
def drop_features(data):
    features_to_drop = ['league_name', 'season', 'shoton', 'shotoff', 'goal', 'corner', 'foulcommit', 'card']
    data.drop(features_to_drop, axis='columns', inplace=True)
    print("features were succssefully droped")
    return data


# This method clean none values
def clean_na(data):
    threshold = int(0.7 * len(data))
    # This loop remove all features that the number of non none values is at least 70%.
    for col in data.columns:
        count = data[col].count()
        if threshold > count:
            print(data[col].count())
            data.drop(col, axis='columns', inplace=True)
    # Clean all row that have more then 5 features with None values.
    data.dropna(thresh=5, inplace=True)
    print("data was succssefully cleaned")
    return data


# This method detects outlier records
def clean_outlier(data):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in data.columns:
        if data[col].dtypes in numerics:
            data = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]

    print("ouliers were succssefully deleted")
    return data


def discritization(data):
    # convert categorial variables into numeric values
    label_encoder = LabelEncoder()
    for category in categorial_vec:
        data[category] = label_encoder.fit_transform(data[category])
    print("data was succssefully discritized")
    return data


# This method drop features from data,
# clean the data from none values and outliers,
# also does a disritization for relevant features
def pre_processing(data):
    data = drop_features(data)
    data = clean_na(data)
    data = clean_outlier(data)
    data = discritization(data)
    # discritization(clean_outlier(clean_na(drop_features(data))))
    print("pre process was succssefully finished")
    return data


# This method creates the goal variable of the data
def create_goal_var(data):
    data['target'] = data.apply(lambda _: '', axis=1)
    for index, row in data.iterrows():
        if row['home_team_goal'] > row['away_team_goal']:
            data['target'] = 1
        elif row['home_team_goal'] < row['away_team_goal']:
            data['target'] = -1
        else:
            data['target'] = 0
    print("target data was succssefully created")
    return data['target']


# This method choose best params model by using grid search algorithm
def best_params_model(model, X_train_, y_train_, grid_variables):
    k_fold = KFold(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size=0.2)
    grid = GridSearchCV(model, grid_variables, scoring='f1', cv=k_fold, refit=True)
    grid_results = grid.fit(X_train, y_train)
    model = grid_results.best_estimator_
    print("model params were succssefully selected")
    return model


# This method adds missing categorials values
def add_categorial_missing_values(data, col, value):
    data[col].fillna(value, inplace=True, axis="columns")


# This method adds missing numerical values
def add_numerical_missing_values(data, col, value):
    data[col].fillna(value, inplace=True)


def fill_nones(train, test):
    # numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in train.columns:
        # if train[col].dtypes in numerics:
        #     mean = train[col].mean()
        #     add_numerical_missing_values(train, col, mean)
        #     add_numerical_missing_values(test, col, mean)
        # else:
        print(train.apply(lambda x: sum(x.isnull()), axis=0))
        best = train[col].mode()[0]
        add_categorial_missing_values(train, col, best)
        print(train.apply(lambda x: sum(x.isnull()), axis=0))
        add_categorial_missing_values(test, col, best)

    print("nones were succssefully imputed")
    print(train)


def fit_model(model, data, X_train_, y_train_):
    k_fold = KFold(n_splits=10)
    f1_max_score = 0
    final_model = None
    for train_index, test_index in k_fold.split(data):
        X_train = X_train_.iloc[train_index, :]
        y_train = y_train_.iloc[train_index]
        X_test = X_train_.iloc[test_index, :]
        y_test = y_train_.iloc[test_index]
        X_train, X_test = fill_nones(X_train, X_test)
        model.fit(X_train, y_train)
        model_prediction = model.predict(X_test)
        f1 = f1_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))

        if f1 > f1_max_score:
            f1_max_score = f1
            final_model = model
    if f1_max_score == 0:
        final_model = model
    print("model was succssefully chosen")
    return final_model


def print_model_evaluation(evaluation_list, name):
    print(f'--------{name}-------')
    print('Accuracy Score: %.3f' % (mean(evaluation_list[0])))
    print('F1 Score: %.3f' % (evaluation_list[1]))


def evaluate_model(model, name, X_test, y_test):
    model_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, model_prediction)
    f1 = f1_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))

    print_model_evaluation([accuracy, f1], name)
    print("model was succssefully evaluated")


# Random Forest
def random_forest(data, X_train, y_train,  X_test, y_test):
    model = RandomForestClassifier()
    n_estimators = [10, 20, 30] + [i for i in range(45, 105, 5)]
    max_depth = [2, 4, 8, 16, 32, 64]
    grid_variables = {'n_estimators': n_estimators, 'max_depth': max_depth}
    best_model = best_params_model(model, X_train, y_train, grid_variables)
    best_model = fit_model(best_model, data, X_train, y_train)
    evaluate_model(best_model, "Random Forest", X_test, y_test)


def handle_data(data):
    # data = xml_change_values(data)
    # clean data (X)
    X = pre_processing(data)
    # create goal variable (y)
    y = create_goal_var(X)
    return X, y


if __name__ == '__main__':
    # handle train data
    X_train, y_train = handle_data(get_train_data())
    data = pd.concat([X_train, y_train], axis=1)

    # handle test data
    X_test, y_test = handle_data(get_test_data())

    random_forest(data, X_train, y_train,  X_test, y_test)

