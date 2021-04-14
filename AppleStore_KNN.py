import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from AppleStore_Milestone2 import *
import time
print('\t\t\t KNeighbors Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
start_t=time.time()
KNeighborsClassifierModel = KNeighborsClassifier(n_neighbors=75)
KNeighborsClassifierModel.fit(X_train, Y_train)
end_t=time.time()

y_trainprediction = KNeighborsClassifierModel.predict(X_train)
accuracyTrain=np.mean(y_trainprediction == Y_train)
print ("The achieved accuracy train using KNN is : " + str(accuracyTrain),'\n training time = : ',end_t-start_t)

start_t=time.time()
y_testprediction = KNeighborsClassifierModel.predict(X_test)
accuracyTest=np.mean(y_testprediction == Y_test)
end_t=time.time()
print ("The achieved accuracy test using KNN is : " + str(accuracyTest),'\n testing time = ',end_t-start_t,'\n')
joblib.dump(KNeighborsClassifierModel,'joblib_KNN_ClassifierModel.pkl')


# start_t=time.time()
# loaded_model = joblib.load('joblib_KNN_ClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy=np.mean(predict == Y_test)
# # accuracy = loaded_model.score(X_test, Y_test)
# end_t=time.time()
# print('KNN_ClassifierModel accuracy test : ' + str(accuracy),'\n time test 2= ',end_t-start_t)
'''
error = []

# Calculating error for K values between 1 and 40
for i in range(65, 75):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(65, 75), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
'''

# loaded_model = joblib.load('joblib_KNN_ClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')