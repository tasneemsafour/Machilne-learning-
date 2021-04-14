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
import joblib
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
import time
#clf = tree.DecisionTreeClassifier(max_depth=4,)

print('\t\t\t\t Decission Tree Classifier Model \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

start_t=time.time()
DecisionTreeClassifierModel =tree.DecisionTreeClassifier(max_depth=4)
DecisionTreeClassifierModel = DecisionTreeClassifierModel.fit(X_train, Y_train)
end_t=time.time()
"""tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=,
#             max_features=None, max_leaf_nodes=10, min_samples_leaf=5,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='random')
#             #criterion = "gini", max_leaf_nodes = 7, min_samples_leaf = 4,
# """
Y_trainPred=DecisionTreeClassifierModel.predict(X_train)


print('MSE of train = ',np.sqrt(mean_squared_error(Y_train,Y_trainPred)))
accuracy = DecisionTreeClassifierModel.score(X_train, Y_train)
print('Decission tree accuracy train: ' + str(accuracy),'\n training time = ',end_t-start_t,'\n')

start_t=time.time()
Y_testPred=DecisionTreeClassifierModel.predict(X_test)
accuracy = DecisionTreeClassifierModel.score(X_test, Y_test)
end_t=time.time()
print('MSE of test = ',np.sqrt(mean_squared_error(Y_test,Y_testPred)))
print('Decission tree accuracy test : ' + str(accuracy),'\n testing time = ',end_t-start_t,'\n')
joblib.dump(DecisionTreeClassifierModel,'joblib_DecisionTreeClassifierModel.pkl')

# loaded_model = joblib.load('joblib_DecisionTreeClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# print('MSE of test = ',np.sqrt(mean_squared_error(Y_test,predict)))
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')


max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
 dtree = tree.DecisionTreeClassifier(criterion='gini', max_depth=i)
 dtree.fit(X_train, Y_train)
 pred = dtree.predict(X_test)
 acc_gini.append(dtree.score(X_test, Y_test))

 dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i)
 dtree.fit(X_train, Y_train)
 pred = dtree.predict(X_test)
 acc_entropy.append(dtree.score(X_test, Y_test))
 ####
 max_depth.append(i)
# visualizing changes in parameters
print('changing of max depth \n',max_depth)
print('changing of information gain \n',acc_gini)
print('changing of entropy \n',acc_entropy)

plt.plot(max_depth,acc_gini, label='gini')
plt.plot(max_depth,acc_entropy, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()



"""
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)
"""