import numpy as np
import pandas as pd
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from AppleStore_PreProcessing import *
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import _column_transformer



def create_preprocessing_reg(data):
  #drop currency col (constant null values ) or useless columns  ** DROP CURRENCY COLUMN BEFORE ROWS DROPPING TO MINIMIZE NUMBER OF ROWS LOSS **
  drop_cols=('currency','vpp_lic','track_name','id')
  data=drop_columns(data,drop_cols)

  #drop null rows after droping columns

  data=drop_row(data)

  #print(data)
  HotEncoder_Cols= 'cont_rating'
  data=OneHot_Encoder(data, HotEncoder_Cols);


  LabelEncoder_Cols=('ver','prime_genre')


  data=label_encoder(data, LabelEncoder_Cols);
  #print(data)
  return data

#select the features
def createXDataReg(X, X_list, data):
    X_list = data.iloc[:, :data.shape[1] - 1]  # all data from size to num_of_lang
    #print(X_list.iloc[0,:])

    # feature scaling (normalize features )

    X_list = featureScaling(np.array(X_list))

    X = pd.DataFrame(X_list[:, :])  # all data from size to num_of_lang after dropping (currency and vpp_lic features)

    return X, X_list



#create Y
def createYDataReg(Y, Y_list, data):
    Y_list = data.iloc[:, data.shape[1] - 1:data.shape[1]]

    # feature scaling y
    Y_list = featureScaling(np.array(Y_list))
    # user_rating
    Y = pd.DataFrame(Y_list[:, 0])
    return Y,Y_list




AppleStore_data = pd.read_csv('AppleStore_training.csv')
AppleStore_data=create_preprocessing_reg(AppleStore_data)
#AppleStore_data=data.iloc[:,:]
#print(AppleStore_data.iloc[0,:9])


'''
#get correlation between the features
corr = AppleStore_data.corr()
#select 10% Correlation training features with the user_rating
top_features = corr.index[abs(corr['user_rating']>0.1)]
#print(top_features)
#Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = AppleStore_data[top_features].corr()
#print(top_corr)
#sns.heatmap(top_corr, annot=True)
#plt.show()
#print('X selected features : \n',AppleStore_data.iloc[0,AppleStore_data.shape[0]+1:AppleStore_data.shape[0]+4])
#print('data pefore scaling \n',AppleStore_data,'\n')
#print(AppleStore_data.iloc[0,:])
'''
X_list=[]
X=pd.DataFrame()
X,X_list=createXDataReg(X, X_list, AppleStore_data)

Y_list=[]
Y=pd.DataFrame()
Y,Y_list=createYDataReg(Y, Y_list, AppleStore_data)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)

#make KFold cross validation for more accurate learning
kf3 = KFold(n_splits=3,shuffle=False)


X_train=[]
X_test=[]

Y_train=[]
Y_test= []
for train_index, test_index in kf3.split(X_list):
    X_train, X_test = X_list[train_index], X_list[test_index]
    Y_train, Y_test = Y_list[train_index], Y_list[test_index]

train_temp = []
#convert Y_train to vector
for array in Y_train:
 for x in array:
    train_temp.append(x)
Y_train=train_temp


test_temp = []
#convert Y_test to vector
for array in Y_test:
 for x in array:
    test_temp.append(x)
Y_test=test_temp
#print(np_arrays)
#print("X_train\n",X_train)
#print("X_test\n",X_test)

#print("Y_train\n",Y_train)
#print("Y_test\n",Y_test)
#print('X-----------:\n',X)
#print('Y-----------:\n',Y)


'''
newtestdata=pd.read_csv('AppleStore_training.csv')
newtestdata=create_preprocessing_reg(newtestdata)
Y_newtest=[]
Y_NTF=pd.DataFrame()
Y_NTF,Y_newtest=createYDataReg(Y_NTF,Y_newtest, newtestdata)
#Y_newtest=np.array(Y_newtest)
X_newtest=[]
X_NTF=pd.DataFrame()
X_NTF,X_newtest=createXDataReg(X_NTF,X_newtest, newtestdata)
X_newtest=np.array(X_newtest)

newtest_temp = []

for array in Y_newtest:
 for x in array:
    newtest_temp.append(x)
Y_newtest=newtest_temp
\'''
print('new X data frame\n',X_NTF)
print('new X list\n',X_newtest)
print('new Y data frame\n',Y_NTF)
print('new Y list\n',Y_newtest)\'''
'''
