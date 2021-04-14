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



#Forset tree nonlinear regression
print('\t\t\t Random Forest Regressor Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
RandomForestRegressorModel.fit(X_train, Y_train)
pred_train_rf= RandomForestRegressorModel.predict(X_train)
print('Mean Square Error of traning data = ',np.sqrt(mean_squared_error(Y_train,pred_train_rf)))
print('aaccuracy of traning data = ',r2_score(Y_train, pred_train_rf))

pred_test_rf = RandomForestRegressorModel.predict(X_test)


print('Mean Square Error of testing data = ',np.sqrt(mean_squared_error(Y_test,pred_test_rf)))
print('aaccuracy of testing data = ',r2_score(Y_test, pred_test_rf),'\n')
joblib.dump(RandomForestRegressorModel,'joblib_RandomForestRegressorModel.pkl')






