from AppleStore_Milestone_cop_1 import *
'''from AppleStore_DecissionTreeRegressor import *
from AppleStore_RandomForest_Regression import *
from AppleStore_Ridge_Polynomial_Regression import *
from AppleStore_MLP_Regrission import *'''
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from AppleStore_RunNewTestDataRegressor import *
import joblib

joblib_file_reg=['joblib_RandomForestRegressorModel.pkl',
                 'joblib_DecisionTreeRegressorModel.pkl',
                 'joblib_MLPRegressorModel.pkl'
                 ]

# Save RL_Model to file in the current working directory
'''joblib.dump(RandomForestRegressorModel, joblib_file_reg[0])
joblib.dump(DecisionTreeRegressorModel, joblib_file_reg[1])
joblib.dump(MLPRegressorModel , joblib_file_reg[2])'''

'''
joblib_file_reg =np.append(joblib_file_reg, ["joblib_PolynomialRidgeModel.pkl"], axis=0)
joblib.dump(PolynomialRidgeModel, joblib_file_reg[2])
'''

print('\t\t\t\t Reload & Test New Data File \t\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
ModelName=['RandomForestRegressor',
           'DecisionTreeRegressor',
           'MLPRegressor']

# Load the models from file
for i in range (len(joblib_file_reg)):
    loaded_model=joblib.load(joblib_file_reg[i])
    predict = loaded_model.predict(X_newtest)
    print('MSE of new test model of {} = {}'.format(ModelName[i],np.sqrt(mean_squared_error(Y_newtest, predict))))
    print('the  accuracy of new test model of {} = {}\n'.format(ModelName[i],r2_score(Y_newtest, predict)))



