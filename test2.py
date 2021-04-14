from AppleStore_Milestone2 import *
from AppleStore_RunNewTestDataClassifier import *
import joblib

loaded_model = joblib.load('joblib_DecisionTreeClassifierModel.pkl')
predict = loaded_model.predict(X_newtest)
accuracy = loaded_model.score(X_newtest, Y_newtest)
print('Accuracy of new test DecisionTreeClassifierModel' , accuracy,'\n')



loaded_model = joblib.load('joblib_RandomForestClassifierModel.pkl')
predict = loaded_model.predict(X_newtest)
accuracy = loaded_model.score(X_newtest, Y_newtest)
print('Accuracy of new test RandomForestClassifierModel' , accuracy,'\n')