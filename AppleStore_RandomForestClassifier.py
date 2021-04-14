import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import *
from AppleStore_Milestone2 import *
import time
print('\t\t\t\t Random Forest Classifier Model \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
start_t=time.time()
RandomForestClassifierModel = RandomForestClassifier(n_estimators = 9,random_state=0)
RandomForestClassifierModel.fit(X_train, Y_train)
end_t=time.time()

y_predtrain = RandomForestClassifierModel.predict(X_train)
accuracy = np.mean(y_predtrain == Y_train)
print('Random forest accuracy train with prediction : ',accuracy,'\n trainning time = ',end_t-start_t)

# Predicting the Test set results
start_t=time.time()
y_predtest = RandomForestClassifierModel.predict(X_test)
accuracy = np.mean(y_predtest == Y_test)
end_t=time.time()
print('Random forest accuracy test with prediction : ',accuracy,'\n testing time = ',end_t-start_t)

joblib.dump(RandomForestClassifierModel,'joblib_RandomForestClassifierModel.pkl')

# loaded_model = joblib.load('joblib_RandomForestClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')