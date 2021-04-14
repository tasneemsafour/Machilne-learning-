from AppleStore_Milestone2 import *
from sklearn import tree
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import time


print('\t\t\tSVM Classifier Model\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
# we create an instance of SVM and fit out data.
C = 0.1  # SVM regularization parameter
start_t1=time.time()
LinearSVModel = svm.LinearSVC(C=C).fit(X_train, Y_train) #minimize squared hinge loss, One vs All
end_t1=time.time()

start_t2=time.time()
NonLinearSVModel = svm.SVC(kernel='rbf', gamma=0.5, C=C).fit(X_train, Y_train)
end_t2=time.time()

start_t3=time.time()
PolynomialSVModel = svm.SVC(kernel='poly', degree=2, C=C).fit(X_train, Y_train)
end_t3=time.time()
arr_t=[]
arr_t.append(end_t1-start_t1)
arr_t.append(end_t2-start_t2)
arr_t.append(end_t3-start_t3)

joblib.dump(LinearSVModel,'joblib_OneVSAllLinearSVModel.pkl')
joblib.dump(NonLinearSVModel,'joblib_NonLinearSVModel.pkl')
joblib.dump(PolynomialSVModel,'joblib_PolynomialSVModel.pkl')
# loaded_model = joblib.load('joblib_PolynomialSVModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')
title=''
for i, clf in enumerate((LinearSVModel, NonLinearSVModel, PolynomialSVModel)):

    if clf==LinearSVModel:
        title='Linear SVM Model'
    elif clf==NonLinearSVModel:
        title='NON Linear SVM Model'
    else:
        title = 'Polynomial SVM Model'
    predictions = clf.predict(X_train)
    accuracy = np.mean(predictions == Y_train)
    print('{} accuracy of train data is = '.format(title), accuracy,'\n training time of {} model = '.format(title),arr_t[i])
    start_t=time.time()
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == Y_test)
    end_t=time.time()
    print('{} accuracy of test data is = '.format(title),accuracy,'\n testing time of {} modet = '.format(title),end_t-start_t,'\n')
print('\n')


