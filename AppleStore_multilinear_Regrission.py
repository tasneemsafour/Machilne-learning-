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

print('\t\t\t Multi Linear Regressor Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
#multilinear regression
MultiLinearModel = linear_model.LinearRegression()
MultiLinearModel.fit(X_train, Y_train)

prediction= MultiLinearModel.predict(X_train)
print('Mean Square Error of train', metrics.mean_squared_error(np.asarray(Y_train), prediction))

print("train accuracy = ",metrics.r2_score(Y_train, prediction))

prediction= MultiLinearModel.predict(X_test)

for i in range (len(Y_test)):
    print('test value',Y_test[i])
    print('pred test',prediction[i])
    print('\n')
print('Mean Square Error of test', metrics.mean_squared_error(np.asarray(Y_test), prediction))