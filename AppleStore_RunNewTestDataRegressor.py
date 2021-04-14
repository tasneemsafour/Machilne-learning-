import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import _column_transformer
from AppleStore_PreProcessing import *
from AppleStore_Milestone_cop_1 import *
import joblib


newtestdata=pd.read_csv('AppleStore_training.csv')
# drop currency col (constant null values ) or useless columns ** DROP CURRENCY COLUMN BEFORE ROWS DROPPING TO MINIMIZE NUMBER OF ROWS LOSS **
drop_cols=('currency','vpp_lic','track_name','id')
newtestdata=drop_columns(newtestdata, drop_cols)
#drop null rows
newtestdata=drop_row(newtestdata)

Y_newtest=[]
Y_NTF=pd.DataFrame()
#creat Y dataframe and list
Y_NTF,Y_newtest=createYDataReg(Y_NTF,Y_newtest, newtestdata)

#load label_encoderModel
cols=('ver','prime_genre')
label_encoderModel=joblib.load('joblib_label_encoderModel.pkl')
for col in cols:
 newtestdata.replace(label_encoderModel[col], inplace=True)

#load Hot_encoderModel
hot_encoderModel=joblib.load('joblib_hot_encoderModel.pkl')
newtestdata =hot_encoderModel.transform(newtestdata)

X_newtest=[]
X_NTF=pd.DataFrame()
#creat X dataframe and list
X_NTF,X_newtest=createXDataReg(X_NTF,X_newtest, newtestdata)
X_newtest=np.array(X_newtest)

newtest_temp = []
#convert Y to vector (if is not a vector error when loading the regression models )
for array in Y_newtest:
 for x in array:
    newtest_temp.append(x)
Y_newtest=newtest_temp

'''
print('new X data frame\n',X_NTF)
print('new X list\n',X_newtest)
print('new Y data frame\n',Y_NTF)
print('new Y list\n',Y_newtest)'''
