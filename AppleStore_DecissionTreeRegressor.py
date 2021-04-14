from sklearn import linear_model
#from AppleStore_Milestone1 import *
from sklearn import metrics

from AppleStore_Milestone_cop_1 import *
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

print('\t\t\t\t Decision Tree Regressor Model \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

DecisionTreeRegressorModel = DecisionTreeRegressor(max_depth=5)

DecisionTreeRegressorModel.fit(X_train, Y_train)

tr2 = DecisionTreeRegressorModel.predict(X_train)

y2 = DecisionTreeRegressorModel.predict(X_test)

print('MSE OF TRAIN = ',np.sqrt(mean_squared_error(Y_train,tr2)))
print('accuracy of train = ',r2_score(Y_train, tr2))

print('MSE OF test = ',np.sqrt(mean_squared_error(Y_test,y2)))
print('accuracy of test = ',r2_score(Y_test, y2),'\n')

joblib.dump(DecisionTreeRegressorModel,'joblib_DecisionTreeRegressorModel.pkl')