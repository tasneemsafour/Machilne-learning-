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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
import time

print('\t\t\t AdaBoost Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
start_t=time.time()
AdaBoostClassifierModel = AdaBoostClassifier(SGDClassifier(alpha=0.01),
                                             algorithm="SAMME",
                                             n_estimators=100)

AdaBoostClassifierModel.fit(X_train, Y_train)
end_t=time.time()


y_trainprediction = AdaBoostClassifierModel.predict(X_train)
accuracyTrain=np.mean(y_trainprediction == Y_train)
print ("The achieved accuracy using Adaboost is " + str(accuracyTrain),'\n trainning time = ',end_t-start_t)

start_t=time.time()
y_testprediction = AdaBoostClassifierModel.predict(X_test)
accuracyTest=np.mean(y_testprediction == Y_test)
end_t=time.time()
print ("The achieved accuracy using Adaboost is " + str(accuracyTest),'\n testining time = ',end_t-start_t)
joblib.dump(AdaBoostClassifierModel,'joblib_AdaBoostClassifierModel.pkl')

# loaded_model = joblib.load('joblib_AdaBoostClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')






#bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
#                         algorithm="SAMME",
#                         n_estimators=300)
#scaler = StandardScaler()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
