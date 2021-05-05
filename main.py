import json
import math
import time

import shap
import xmltodict
import sys
# from bs4 import BeautifulSoup
import sqlite3
import pandas as pd
import numpy as np
from sklearn import metrics, svm
from sklearn.preprocessing import LabelEncoder
from numpy.core import mean
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn import tree

# imports for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



categorial_vec = ['league_name', 'season']
numerical_vec = ['stage', 'home_team_goal', 'away_team_goal']


# get train data from sqlite
def get_train_data():
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
                                            HTeam.team_api_id AS home_team_id,
                                            ATeam.team_api_id AS away_team_id,
                                            home_team_goal, 
                                            away_team_goal,
                                            home_player_1, 
                                            home_player_2,
                                            home_player_3, 
                                            home_player_4, 
                                            home_player_5, 
                                            home_player_6, 
                                            home_player_7, 
                                            home_player_8, 
                                            home_player_9, 
                                            home_player_10, 
                                            home_player_11, 
                                            away_player_1, 
                                            away_player_2, 
                                            away_player_3, 
                                            away_player_4, 
                                            away_player_5, 
                                            away_player_6, 
                                            away_player_7, 
                                            away_player_8, 
                                            away_player_9, 
                                            away_player_10, 
                                            away_player_11
                                    FROM Match
                                    JOIN League on League.id = Match.league_id
                                    LEFT JOIN Team AS HTeam on HTeam.team_api_id = Match.home_team_api_id
                                    LEFT JOIN Team AS ATeam on ATeam.team_api_id = Match.away_team_api_id
                                    WHERE season not like '2015/2016' and goal is not null
                                    LIMIT 100
                                   ;""", conn)

    print("Got train data succssefully")
    return data


def get_player_attributes():
    conn = sqlite3.connect('database.sqlite')
    players_data = pd.read_sql("""SELECT player_api_id,date, overall_rating, potential, stamina
                        FROM Player_Attributes""", conn)
    # avg all records for each player
    grouped_players_data = players_data.groupby('player_api_id').agg('mean').round(0)
    print("Got player data succssefully")
    return grouped_players_data


def get_test_data():
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
                                            HTeam.team_api_id AS home_team_id,
                                            ATeam.team_api_id AS away_team_id,
                                            home_team_goal, 
                                            away_team_goal,
                                            home_player_1, 
                                            home_player_2,
                                            home_player_3, 
                                            home_player_4, 
                                            home_player_5, 
                                            home_player_6, 
                                            home_player_7, 
                                            home_player_8, 
                                            home_player_9, 
                                            home_player_10, 
                                            home_player_11, 
                                            away_player_1, 
                                            away_player_2, 
                                            away_player_3, 
                                            away_player_4, 
                                            away_player_5, 
                                            away_player_6, 
                                            away_player_7, 
                                            away_player_8, 
                                            away_player_9, 
                                            away_player_10, 
                                            away_player_11
                                    FROM Match
                                    JOIN League on League.id = Match.league_id
                                    LEFT JOIN Team AS HTeam on HTeam.team_api_id = Match.home_team_api_id
                                    LEFT JOIN Team AS ATeam on ATeam.team_api_id = Match.away_team_api_id
                                    WHERE season like '2015/2016' and goal is not null
                                    ORDER by date
                                    LIMIT 100;
                                    """, conn)


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

        if feature == "shoton":
            handle_shot_on(data)
            data = drop_features(data, ["shoton"])

        elif feature == "shotoff":
            handle_shot_off(data)
            data = drop_features(data, ["shotoff"])

        elif feature == "goal":
            # handle_goal(data)
            data = drop_features(data, ["goal"])

        elif feature == "corner":
            handle_corner(data)
            data = drop_features(data, ["corner"])

        elif feature == "foulcommit":
            handle_foulcommit(data)
            data = drop_features(data, ["foulcommit"])

        elif feature == "card":
            handle_card(data)
            data = drop_features(data, ["card"])

    print("xml data was succssefully converted")
    return data


# This method drop unnecessary features
def drop_features(data, features_to_drop):
    data.drop(features_to_drop, axis='columns', inplace=True)
    print("features were succssefully droped" + str(features_to_drop))
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
    data.dropna(thresh=7, inplace=True)
    print("data was succssefully cleaned")
    return data


# This method detects outlier records
# TODO: change numeric into columns names
def clean_outlier(data):
    for col in numerical_vec:
        data[col] = data[np.abs(data[col]-data[col].mean()) <= (3*data[col].std())]
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
    data = clean_na(data)

    #TODO: problems in clean_outlier(data), discritization(data)
    #data = clean_outlier(data)
    data = discritization(data)
    data = get_players_statistics(data, get_player_attributes())

    players = []
    for i in range(11):
        players.append('home_player_' + str(i+1))
        players.append('away_player_' + str(i+1))

    data = drop_features(data, players)

    # discritization(clean_outlier(clean_na(drop_features(data))))
    print("train data Nulls")
    print(data.apply(lambda x: sum(x.isnull()), axis=0))
    print("pre process was succssefully finished")
    return data


# This method creates the goal variable of the data
#TODO: change types
def create_goal_var(data):
    # data['target'] = data.apply(lambda _: 0, axis=1)
    for index, row in data.iterrows():
        if data.iloc[index]['home_team_goal'] > data.iloc[index]['away_team_goal']:
            data.loc[index, 'target'] = 1
        elif data.iloc[index]['home_team_goal'] < data.iloc[index]['away_team_goal']:
            data.loc[index, 'target'] = -1
        else:
            data.loc[index, 'target'] = 0
    print("target data was succssefully created")
    data = drop_features(data, ["home_team_goal", "away_team_goal", "id", "season", "home_team_id","away_team_id"])
    return data


# This method choose best params model by using grid search algorithm
# def best_params_model(model, X_train_, y_train_, grid_variables):
#     k_fold = KFold(n_splits=5)
#     X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size=0.2)
#     grid = GridSearchCV(model, grid_variables, scoring='f1', cv=k_fold, refit=True)
#     grid_results = grid.fit(X_train, y_train)
#     model = grid_results.best_estimator_
#     print("model params were succssefully selected")
#     return model

def best_params_model(model, X_train_, y_train_, grid_variables):
    k_fold = KFold(n_splits=5)
    X_train, X_test, y_train, y_test = train_test_split(X_train_, y_train_, test_size=0.2)
    grid = GridSearchCV(model, grid_variables, cv=k_fold)
    grid_results = grid.fit(X_train, y_train)
    model = grid_results.best_estimator_
    print("model params were succssefully selected")
    return model

def classification_model(model,x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model.fit(X_train,y_train)

    model_prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,model_prediction)

    print("cross validation accuracy score %s" % "{0:.3}".format(accuracy))
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
    k_fold = KFold(n_splits=5)
    f1_max_score = 0
    final_model = None
    for train_index, test_index in k_fold.split(data):
        X_train = X_train_.iloc[train_index, :]
        y_train = y_train_.iloc[train_index]
        X_test = X_train_.iloc[test_index, :]
        y_test = y_train_.iloc[test_index]
        # TODO: Check this
        # X_train, X_test = fill_nones(X_train, X_test)
        model.fit(X_train, y_train)
        model_prediction = model.predict(X_test)
        f1 = f1_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))

        if f1 > f1_max_score:
            f1_max_score = f1
            final_model = model

    print("model was succssefully chosen")
    return final_model


def print_model_evaluation(evaluation_list, name):
    print(f'--------{name}-------')
    print('Accuracy Score: %.3f' % (mean(evaluation_list[0])))
    print('F1 Score: %.3f' % (evaluation_list[1]))
    print('Precision Score: %.3f' % (evaluation_list[2]))
    print('Recall Score: %.3f' % (evaluation_list[3]))


def evaluate_model(model, name, X_test, y_test):
    model_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, model_prediction)
    f1 = f1_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))
    precision = precision_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))
    recall = recall_score(y_test, model_prediction, average='weighted', labels=np.unique(model_prediction))

    print_model_evaluation([accuracy, f1, precision, recall], name)
    print("model was succssefully evaluated")


# Random Forest
def random_forest(data, X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    n_estimators = [10, 20, 30] + [i for i in range(45, 105, 5)]
    max_depth = [2, 4, 8, 16, 32, 64]
    grid_variables = {'n_estimators': n_estimators, 'max_depth': max_depth}
    best_model = best_params_model(model, X_train, y_train, grid_variables)
    best_model = fit_model(best_model, data, X_train, y_train)
    evaluate_model(best_model, "Random Forest", X_test, y_test)
    # features_names = list(X_train.columns.values)
    # features_importance(model, features_names, "Random Forest")
    shap_feature_importance(best_model, X_train, False)

# Logistic Regression
def logistic_regression(data, X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    solver = ["newton-cg", "liblinear"]
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    random_state = [None, 0, 42]
    max_iter = [600, 700]
    grid_variables = {'C': C, 'random_state': random_state, 'solver': solver, 'max_iter': max_iter}
    best_model = best_params_model(model, X_train, y_train, grid_variables)
    best_model = fit_model(best_model, data, X_train, y_train)
    # model.fit(X_train, y_train)
    evaluate_model(best_model, "Logistic Regression", X_test, y_test)

# CART
def CART(data, X_train, y_train, X_test, y_test):
    model = tree.DecisionTreeClassifier()
    max_depth = [2, 4, 8, 16, 32, 64]
    min_weight_fraction_leaf = [i/20 for i in range(0, 10)]
    grid_variables = {'max_depth': max_depth, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    best_model = best_params_model(model, X_train, y_train, grid_variables)
    best_model = fit_model(best_model, data, X_train, y_train)
    evaluate_model(best_model, "CART", X_test, y_test)

def svm_model(X_train, y_train, X_test, y_test):

    clf = svm.SVC()
    kernel = ['poly', 'rbf', 'sigmoid']
    coef0 = [i/4 for i in range(0, 16)]
    grid_variables = {'kernel': kernel, 'coef0': coef0}
    best_model = best_params_model(clf, X_train, y_train, grid_variables)

    # Train the model using the training sets
    best_model.fit(X_train, y_train)

    evaluate_model(best_model, str(best_model.kernel) + " SVM", X_test, y_test)


def get_players_statistics(data, players_data):

    features_names = list(players_data.columns.values)

    for index, row in data.iterrows():

        lst_home = [0] * len(features_names)
        lst_away = [0] * len(features_names)

        num_h_players = 11
        num_a_players = 11
        for i in range(11):
            home_player_id = data.iloc[index]['home_player_' + str(i+1)]
            if math.isnan(home_player_id):
                num_h_players -= 1
            else:
                for idx, feature in enumerate(features_names):
                    lst_home[idx] += players_data.at[home_player_id, feature]

            away_player_id = data.iloc[index]['away_player_' + str(i+1)]
            if math.isnan(away_player_id):
                num_a_players -= 1
            else:
                for idx, feature in enumerate(features_names):
                    lst_away[idx] += players_data.at[away_player_id, feature]

            home_str = "home_team_"
            away_str = "away_team_"


        for idx,feature in enumerate(features_names):
            try:
                data.loc[index, home_str + feature] = (lst_home[idx] / num_h_players).round(0)
                data.loc[index, away_str + feature] = (lst_away[idx] / num_a_players).round(0)
            except:
                print((lst_home[idx] / num_h_players))
                print((lst_home[idx] / num_h_players))

    return data


def handle_shot_on(data):

    ## iterate over the data table after the xml function turend each col into json string

    for index,rows in data.iterrows():

        ## counters for  each col

        home_team_shoot_on = 0
        away_team_shoot_on = 0

        ## this row change to value of the json string into a dict or a list of dicts depends on how many
        ## shot on record there was in the match if we had one event its a dict,else its a list of dicts

        try:
            curr_dict = json.loads(data.iloc[index]["shoton"])["shoton"]["value"]

        except:
            data.loc[index, 'home_shoton'] = float(0)
            data.loc[index, 'away_shoton'] = float(0)
            continue

        ## if it a list of dict

        if type(curr_dict) is list:

            #iterate over the list
            for shoton in curr_dict:

                ## check if the team value in the xml dict match the home id col or the away to
                ## update the counters accordingly
                try:

                    if float(shoton["team"]) == data.iloc[index]['home_team_id']:
                        home_team_shoot_on += float(1)
                    elif float(shoton["team"]) == data.iloc[index]['away_team_id']:
                        away_team_shoot_on += float(1)
                except:
                    continue
            ## set the new col of the home and away shot in the data dataframe
            data.loc[index, 'home_shoton'] = home_team_shoot_on
            data.loc[index, 'away_shoton'] = away_team_shoot_on

        ## if it a single dict
        else:
            ## same logic guest without a loop
            try:
                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_shoot_on += float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_shoot_on += float(1)

                data.loc[index, 'home_shoton'] = home_team_shoot_on
                data.loc[index, 'away_shoton'] = away_team_shoot_on

            except:
                data.loc[index, 'home_shoton'] = float(0)
                data.loc[index, 'away_shoton'] = float(0)
                continue


def handle_shot_off(data):

    for index,rows in data.iterrows():

        home_team_shoot_off = 0
        away_team_shoot_off = 0

        try:
            curr_dict = json.loads(data.iloc[index]["shotoff"])["shotoff"]["value"]

        except:
            data.loc[index, 'home_shotoff'] = float(0)
            data.loc[index, 'away_shotoff'] = float(0)
            continue

        if type(curr_dict) is list:

            for shotoff in curr_dict:

                try:

                    if float(shotoff["team"]) == data.iloc[index]['home_team_id']:
                        home_team_shoot_off -= float(1)
                    elif float(shotoff["team"]) == data.iloc[index]['away_team_id']:
                        away_team_shoot_off -= float(1)

                except:
                    continue

            ## set the new col of the home and away shot in the data dataframe
            data.loc[index, 'home_shotoff'] = home_team_shoot_off
            data.loc[index, 'away_shotoff'] = away_team_shoot_off

        else:

            try:
                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_shoot_off -= float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_shoot_off -= float(1)

                data.loc[index, 'home_shotoff'] = home_team_shoot_off
                data.loc[index, 'away_shotoff'] = away_team_shoot_off

            except:
                data.loc[index, 'home_shotoff'] = float(0)
                data.loc[index, 'away_shotoff'] = float(0)
                continue

def handle_goal(data):

    for index,rows in data.iterrows():

        home_team_goal = 0
        away_team_goal = 0

        try:
            curr_dict = json.loads(data.iloc[index]["goal"])["goal"]["value"]

        except:
            data.loc[index, 'home_goal'] = float(0)
            data.loc[index, 'away_goal'] = float(0)
            continue

        if type(curr_dict) is list:

            for goal in curr_dict:

                try:
                    if float(goal["team"]) == data.iloc[index]['home_team_id']:
                        home_team_goal += float(1)
                    elif float(goal["team"]) == data.iloc[index]['away_team_id']:
                        away_team_goal += float(1)
                except:
                    continue

            ## set the new col of the home and away shot in the data dataframe
            data.loc[index, 'home_goal'] = home_team_goal
            data.loc[index, 'away_goal'] = away_team_goal

        else:
            try:

                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_goal += float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_goal += float(1)

                data.loc[index, 'home_goal'] = home_team_goal
                data.loc[index, 'away_goal'] = away_team_goal

            except:
                data.loc[index, 'home_goal'] = float(0)
                data.loc[index, 'away_goal'] = float(0)
                continue


def handle_corner(data):

    for index,rows in data.iterrows():

        home_team_corner = 0
        away_team_corner = 0

        try:
            curr_dict = json.loads(data.iloc[index]["corner"])["corner"]["value"]

        except:
            data.loc[index, 'home_corner'] = float(0)
            data.loc[index, 'away_corner'] = float(0)
            continue

        if type(curr_dict) is list:

            for goal in curr_dict:
                try:
                    if float(goal["team"]) == data.iloc[index]['home_team_id']:
                        home_team_corner += float(1)
                    elif float(goal["team"]) == data.iloc[index]['away_team_id']:
                        away_team_corner += float(1)

                except:
                    continue

            ## set the new col of the home and away shot in the data dataframe
            data.loc[index, 'home_corner'] = home_team_corner
            data.loc[index, 'away_corner'] = away_team_corner

        else:
            try:
                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_corner += float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_corner += float(1)

                data.loc[index, 'home_corner'] = home_team_corner
                data.loc[index, 'away_corner'] = away_team_corner

            except:
                data.loc[index, 'home_corner'] = float(0)
                data.loc[index, 'away_corner'] = float(0)
                continue

def handle_card(data):

    for index, rows in data.iterrows():

        home_team_cards = 0
        away_team_cards = 0

        try:
            curr_dict = json.loads(data.iloc[index]["card"])["card"]["value"]

        except:
            data.loc[index, 'home_cards'] = float(0)
            data.loc[index, 'away_cards'] = float(0)
            continue

        if type(curr_dict) is list:

            for goal in curr_dict:
                try:
                    if float(goal["team"]) == data.iloc[index]['home_team_id']:
                        home_team_cards -= float(1)
                    elif float(goal["team"]) == data.iloc[index]['away_team_id']:
                        away_team_cards -= float(1)

                except:
                    continue

            ## set the new col of the home and away shot in the data dataframe
            data.loc[index, 'home_cards'] = home_team_cards
            data.loc[index, 'away_cards'] = away_team_cards

        else:
            try:
                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_cards -= float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_cards -= float(1)

                data.loc[index, 'home_cards'] = home_team_cards
                data.loc[index, 'away_cards'] = away_team_cards

            except:
                data.loc[index, 'home_cards'] = float(0)
                data.loc[index, 'away_cards'] = float(0)
                continue


def handle_foulcommit(data):

    for index, rows in data.iterrows():

        home_team_fouls = 0
        away_team_fouls = 0

        try:
            curr_dict = json.loads(data.iloc[index]["foulcommit"])["foulcommit"]["value"]

        except:
            data.loc[index, 'home_fouls'] = float(0)
            data.loc[index, 'away_fouls'] = float(0)
            continue

        if type(curr_dict) is list:

            for goal in curr_dict:
                try:
                    if float(goal["team"]) == data.iloc[index]['home_team_id']:
                        home_team_fouls -= float(1)
                    elif float(goal["team"]) == data.iloc[index]['away_team_id']:
                        away_team_fouls -= float(1)

                except:
                    continue


            data.loc[index, 'home_fouls'] = home_team_fouls
            data.loc[index, 'away_fouls'] = away_team_fouls

        else:
            try:
                if float(curr_dict["team"]) == data.iloc[index]['home_team_id']:
                    home_team_fouls -= float(1)
                elif float(curr_dict["team"]) == data.iloc[index]['away_team_id']:
                    away_team_fouls -= float(1)

                data.loc[index, 'home_fouls'] = home_team_fouls
                data.loc[index, 'away_fouls'] = away_team_fouls

            except:
                data.loc[index, 'home_fouls'] = float(0)
                data.loc[index, 'away_fouls'] = float(0)
                continue







def shap_feature_importance(model, X_df, one_dimension):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)
    if not one_dimension:
        shap_values.values = shap_values.values[:, :, 1]
        shap_values.base_values = shap_values.base_values[:, 1]
    shap.plots.beeswarm(shap_values, max_display=50)
    shap.plots.bar(shap_values)


def handle_data(data):
    data = xml_change_values(data)
    # clean data (X)
    X = pre_processing(data)
    # create goal variable (y)
    create_goal_var(X)
    X = data.iloc[:, :-1]
    y = data['target']

    return data, X, y



if __name__ == '__main__':
    # handle train data
    train_data, X_train, y_train = handle_data(get_train_data())

    # handle test data (sessons 2015/2016)
    test_data, X_test, y_test = handle_data(get_test_data())

    svm_model(X_train, y_train,  X_test, y_test)
    logistic_regression(train_data, X_train, y_train,  X_test, y_test)
    random_forest(train_data, X_train, y_train,  X_test, y_test)
    CART(train_data, X_train, y_train, X_test, y_test)


