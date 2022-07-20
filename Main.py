#%% Import libraries
import csv
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyparsing import col
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  make_scorer, mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

#%% Functions
def load_data(filename, columns=None):
    data = pd.read_csv(filename)
    return data


raw_data = load_data('data.csv')
X,Y = raw_data.iloc[:,:-1],raw_data.iloc[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8,shuffle=True,random_state=7)
scorer = make_scorer(r2_score, greater_is_better = True)

parameters = ["eta", "n_estimators", "max_depth", "min_child_weight", "subsample", "alpha", "lambda"]
lbs = np.array([0.1, 10, 3, 1, 0.6, 0, 0.5])
ubs = np.array([0.9, 200, 20, 6, 1, 0.0001, 1])

#%% Objective Function
def objective_function(parameters):
    res = []
    for p in parameters:
        score = cross_val_score(
            XGBRegressor(
            eta=p[0],
            n_estimators=int(p[1]),
            max_depth=int(p[2]),
            min_child_weight=int(p[3]),
            subsample=p[4],
            reg_alpha=p[5],
            reg_lambda=p[6]
            ), X_train, Y_train, scoring=scorer)
        
        res.append(1-np.mean(score))
    return res
#%% Estabilish the MH
# eta, n_estimators, max_depth, min_child_weight, max_delta_step, subsample, alpha, lambda

import pyswarms as ps
bounds = (lbs, ubs)
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO with bounds argument
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=7, options=options, bounds=bounds)
cost, pos = optimizer.optimize(objective_function, iters=100)

print(pos)