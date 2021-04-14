import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import _column_transformer
from AppleStore_PreProcessing import *
from AppleStore_Milestone2 import *
import joblib


newtestdata=pd.read_csv('AppleStore_training_classification.csv')
#get the target of data
target_newtest=newtestdata.iloc[:,newtestdata.shape[1]-1]
#print('target new data \n',target_newtest)

# drop currency col (constant null values ) or useless columns ** DROP CURRENCY COLUMN BEFORE ROWS DROPPING TO MINIMIZE NUMBER OF ROWS LOSS **
drop_cols=('currency','vpp_lic','track_name','id')
newtestdata=drop_columns(newtestdata, drop_cols)

Y_newtest=[]
#create lables for each class
Y_newtest=create_classes(target_newtest,Y_newtest)
Y_newtest=np.array(Y_newtest)
#print('new y',Y_newtest)

#load most frequent model
loaded_model=joblib.load('joblib_imp_mostfreqModel.pkl')
p = loaded_model.transform(newtestdata.iloc[:, 4:7])
#print(newtestdata.iloc[0,:])
newtestdata.iloc[:, 4] = p[:, 0]
newtestdata.iloc[:, 5] = p[:, 1]
newtestdata.iloc[:, 6] = p[:, 2]

#load label_encoderModel
cols=('ver','prime_genre')
label_encoderModel=joblib.load('joblib_label_encoderModel.pkl')
for col in cols:
 newtestdata.replace(label_encoderModel[col], inplace=True)

#load Hot_encoderModel
hot_encoderModel=joblib.load('joblib_hot_encoderModel.pkl')
newtestdata =hot_encoderModel.transform(newtestdata)

#remove the target from the data
newtestdata = newtestdata.iloc[:, :newtestdata.shape[1] - 1]

#load mean model
loaded_model=joblib.load('joblib_imp_mean_dataModel.pkl')
temp_mean = loaded_model.transform(newtestdata.iloc[:, :])
newtestdata.iloc[:, :] = temp_mean

#create feature scaling
newtestdata = featureScaling(np.array(newtestdata))
newtestdata = pd.DataFrame(newtestdata)
#detremine the important features
loaded_model=joblib.load('joblib_imporfeatModel.pkl')
imp_feat=loaded_model.feature_importances_
#print(imp_feat)

#select features
X_newtest=pd.DataFrame()
X_newtest=createXData(X_newtest,newtestdata)
X_newtest=np.array(X_newtest)


