from sklearn.linear_model import LogisticRegression
import numpy as np
from AppleStore_Milestone2 import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
import time

print('\t\t\t Logistic Regression Model mult(1 vs rest)\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
start_t=time.time()
#logistic regression multiclass using one verses rest
LogisticRegressionModel = LogisticRegression(multi_class='ovr', class_weight='auto')
LogisticRegressionModel .fit(X_train, Y_train)
end_t=time.time()

# predictions = LogisticRegressionModel .predict(X_train)
#Generate a confusion matrix
score = LogisticRegressionModel .score(X_train, Y_train)
print("accuracy of train is ",score,'\n trainning time = ',end_t-start_t)

start_t=time.time()
# predictions = LogisticRegressionModel .predict(X_test)
#Generate a confusion matrix
score = LogisticRegressionModel .score(X_test, Y_test)
end_t=time.time()
print("accuracy of test is ",score,'\n testing time = ',end_t-start_t)
joblib.dump(LogisticRegressionModel,'joblib_LogisticRegressionModel.pkl')

# loaded_model = joblib.load('joblib_LogisticRegressionModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')
# print(X_test)
# print(Y_test)
# b = LogisticRegressionModel.intercept_[0]
# w1,w2,w3,w4,w5,w6 = LogisticRegressionModel.coef_.T
# print(w1)
# print(w2)
# print(w3)
# print(w4)
# print(w5)
# print(w6)
#
# # Calculate the intercept and gradient of the decision boundary.
#
#
# # Plot the data and the classification with the decision boundary.
#
# x_values = [np.min(X_test[:, 0] - 5), np.max(X_test[:, 5] + 5)]
# print(x_values)
#
# y_values = - (b + np.dot(w1, x_values)+np.dot(w2, x_values)+np.dot(w3, x_values)+np.dot(w4, x_values)+np.dot(w5, x_values)+np.dot(w6, x_values)) / w6
# plt.plot(x_values, y_values, label='Decision Boundary')
# plt.ylabel(r'$x_2$')
# plt.xlabel(r'$x_1$')
# plt.scatter(X_test[np.where(Y_test == 2),0], X_test[np.where(Y_test == 2),1],X_test[np.where(Y_test == 2),2],X_test[np.where(Y_test == 2),3],X_test[np.where(Y_test == 2),4],X_test[np.where(Y_test == 2),5],s=8, alpha=0.5)
# plt.scatter(X_test[np.where(Y_test == 1),0],X_test[np.where(Y_test == 1),1],X_test[np.where(Y_test == 1),2],X_test[np.where(Y_test == 1),3],X_test[np.where(Y_test == 1),4],X_test[np.where(Y_test == 1),5], s=8, alpha=0.5)
# plt.scatter(X_test[np.where(Y_test == 0),0],X_test[np.where(Y_test == 0),1],X_test[np.where(Y_test == 0),2],X_test[np.where(Y_test == 0),3],X_test[np.where(Y_test == 0),4],X_test[np.where(Y_test == 0),5], s=8, alpha=0.5)
#
# plt.show()