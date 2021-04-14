from AppleStore_Milestone2 import *
'''from AppleStore_DecissionTreeClassifier import *
from AppleStore_RandomForestClassifier import *
from AppleStore_Adaboost import *
from AppleStore_KNN import *
from AppleStore_GradientBoostingClassifier import *
from AppleStore_ExtraTreeClassifier import *
from AppleStore_LogisticRegtession import *
from AppleStore_SVMKernelsClassifier import *'''
from AppleStore_RunNewTestDataClassifier import *
import numpy as np
import joblib
#creat name of file
joblib_file=['joblib_DecisionTreeClassifierModel.pkl',
             'joblib_GradientBoostingClassifierModel.pkl',
             'joblib_RandomForestClassifierModel.pkl',
             'joblib_ExtraTreeClassifierModel.pkl',
             'joblib_AdaBoostClassifierModel.pkl',
             'joblib_KNN_ClassifierModel.pkl',
             'joblib_LogisticRegressionModel.pkl',
             'joblib_OneVSAllLinearSVModel.pkl',
             'joblib_NonLinearSVModel.pkl',
             'joblib_PolynomialSVModel.pkl'

             ]
# Save RL_Model to file in the current working directory
'''
joblib.dump(DecisionTreeClassifierModel,joblib_file[0])
joblib.dump(GradientBoostingClassifierModel, joblib_file[1])
joblib.dump(RandomForestClassifierModel, joblib_file[2])
joblib.dump(ExtraTreeClassifierModel, joblib_file[3])
joblib.dump(AdaBoostClassifierModel, joblib_file[4])
joblib.dump(KNeighborsClassifierModel, joblib_file[5])
joblib.dump(LogisticRegressionModel, joblib_file[6])'''

'''
joblib_file =np.append(joblib_file,["joblib_BaggingClassifierModel.pkl"],axis=0)
joblib.dump(BaggingClassifierModel, joblib_file[6])
'''
'''
joblib_file =np.append(joblib_file,["joblib_Polynomial_SVM_Model.pkl"],axis=0)
joblib.dump(PolynomialSVModel, joblib_file[7])
'''

print('\t\t\t\t Reload & Test New Data File \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
ModelName=[ 'DecisionTreeClassifier',
             'GradientBoostingClassifier',
             'RandomForestClassifier',
             'ExtraTreeClassifier',
             'AdaBoostClassifier',
             'KNN_Classifier',
             'LogisticRegression',

             'OneVSAllLinearSVM',
             'NonLinearSVModel',
             'PolynomialSVM'
             ]
# Load from file and predict new test data
for i in range (len(joblib_file)):
    loaded_model=joblib.load(joblib_file[i])
    predict=loaded_model.predict(X_newtest)
    accuracy = loaded_model.score(X_newtest, Y_newtest)
    print('Accuracy of new test model of {} = {}\n'.format(ModelName[i],accuracy))





