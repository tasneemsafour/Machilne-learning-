import itertools

#from AppleStore_Milestone1 import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_regression
from AppleStore_Milestone_cop_1 import *


print('\t\t\t MLP Regressor Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
MLPRegressorModel = MLPRegressor(hidden_layer_sizes=10, learning_rate_init=0.1, random_state=1, max_iter=550, tol=0.0000001, alpha = 0.000001, solver='lbfgs', n_iter_no_change=10, early_stopping=True)\
    .fit(X_train, Y_train)
prediction_s = MLPRegressorModel.predict(X_train)
print('Mean Square Error train = ', metrics.mean_squared_error(Y_train, prediction_s))
print("Accuracy of MLPRegression train = ", r2_score(Y_train,prediction_s))

prediction_s = MLPRegressorModel.predict(X_test)
print('Mean Square Error test = ', metrics.mean_squared_error(Y_test, prediction_s))

print("Accuracy of MLPRegression test = ", r2_score(Y_test,prediction_s),'\n')

joblib.dump(MLPRegressorModel ,'joblib_MLPRegressorModel.pkl')
###################################linear_model(Lasso)######bad###########################################
'''ls = linear_model.Lasso(alpha=0.000001) #alpha can be fine tuned
ls.fit(X_train, Y_train)
predicted1 = ls.predict(X_test)
print("Lasso Predicted Values")
print (predicted1)
print ('Mean Square Error Lasso')
mse_1 = mean_squared_error(Y_test, predicted1)
print (mse_1)
true_player_value=np.asarray(Y_test)[0]
predicted_player_value=predicted1[0]
print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))
print(ls.score(X_train,Y_train))
###################################linear_model######bad###########################################

'''