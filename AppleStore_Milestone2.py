import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import _column_transformer
from AppleStore_PreProcessing import *
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
import joblib

joblib_preprocfile=[]
def create_classes(targetdata,Ydata):
 for val in targetdata:
    if(val == 'Intermediate'):
        Ydata.append(0)
    elif (val=='High'):
        Ydata.append(1)
    else:
        Ydata.append(-1)
 return Ydata


def createPreprocessingClassi(data):
 #drop currency col (constant null values ) or useless columns ** DROP CURRENCY COLUMN BEFORE ROWS DROPPING TO MINIMIZE NUMBER OF ROWS LOSS **
 drop_cols=('currency','vpp_lic','track_name','id')
 data=drop_columns(data, drop_cols)

 #fill missing values with the most frequent of categorical features
 imp_mostfreqModel = SimpleImputer(strategy='most_frequent')
 imp_mostfreqModel.fit(data.iloc[:,4:7])
 p= imp_mostfreqModel.transform(data.iloc[:,4:7])

 data.iloc[:,4]=p[:,0]
 data.iloc[:,5]=p[:,1]
 data.iloc[:,6]=p[:,2]
 #save most_frequent model
 joblib.dump(imp_mostfreqModel,'joblib_imp_mostfreqModel.pkl')

 #when we try to use only most_frequent without mean model
 '''
 data = data.iloc[:, :data.shape[1] - 1]
 imp_mostfreqModel = SimpleImputer(strategy='most_frequent')
 imp_mostfreqModel.fit(data)
 p = imp_mostfreqModel.transform(data)
 data.iloc[:,:]=p
 print(data.iloc[data.shape[0]-1,:])'''

 #labelEncoder
 LabelEncoder_Cols=('ver','prime_genre')
 data=label_encoder(data, LabelEncoder_Cols);

 #test one hot encoder of 'prime_genre'
 # HotEncoder_Cols = 'prime_genre'
 # data = OneHot_Encoder(data, HotEncoder_Cols);

 #hotEncoder
 HotEncoder_Cols= 'cont_rating'
 data=OneHot_Encoder(data, HotEncoder_Cols);

 data = data.iloc[:, :data.shape[1] - 1]
 #print(data.iloc[0,:])
 #print(data.iloc[:,AppleStore_data.shape[1]-1])
 #print(data.iloc[:,8])
 #print("Size  Price : \n",data.iloc[:,:2])

 #fill missing values with the mean of numerical features
 imp_mean_dataModel = SimpleImputer(missing_values=np.nan, strategy='mean')
 imp_mean_dataModel.fit(data.iloc[:,:])
 temp_mean = imp_mean_dataModel.transform(data.iloc[:,:])
 data.iloc[:,:] = temp_mean
#save mean Model
 joblib.dump(imp_mean_dataModel ,'joblib_imp_mean_dataModel.pkl' )

 '''
 imp_mean_dataModel = imp_mean_dataModel.fit(data.iloc[:, 33:])
 temp_mean = imp_mean_dataModel.transform(data.iloc[:, 33:])
 data.iloc[:, 33:] = temp_mean'''
 #print(data.iloc[data.shape[0]-1,:])
 dataname=data
 # create feature scaling of the data
 data =featureScaling(np.array(data))
 data =pd.DataFrame(data)
 #print(AppleStore_data)
 return data

#select features
def createXData(X, data):

 X['size_bytes']= data.iloc[:, 0]
 X['price']= data.iloc[:, 1]
 X['rating_count_tot']= data.iloc[:, 2]
 X['rating_count_ver ']= data.iloc[:, 3]
 X['ver']= data.iloc[:, 4]
 #take prime_genre as a feature after label encoder but that decrease the accuracy
 #X['prime_genre'] = data.iloc[:, 9]
 X['lang.num']= data.iloc[:, 12]


#make one hot encoder for <--prime_genre--> 2- select the important features 3- select the important features as <--X-->
#but the onehot encoder of <--prime_genre--> decrease the accuracy
 '''
 X['size_bytes']= data.iloc[:, 0]
 X['price']= data.iloc[:, 1]
 X['rating_count_tot']= data.iloc[:, 2]
 X['rating_count_ver ']= data.iloc[:, 3]
 X['ver']= data.iloc[:, 4]
 X['cont_rating_12+'] = data.iloc[:, 6]
 X['cont_rating_17+'] = data.iloc[:, 7]
 X['cont_rating_9+'] = data.iloc[:, 8]
 X['prime_genre_Entertainment'] = data.iloc[:, 22]
 X['prime_genre_Photo & Video'] = data.iloc[:, 23]
 X['sup_devices.num'] = data.iloc[:, 33]
 X['lang.num'] = data.iloc[:, 35]'''

 return X

AppleStore_data= pd.read_csv('AppleStore_training_classification.csv')
target =AppleStore_data.iloc[:, AppleStore_data.shape[1] - 1]
AppleStore_data=createPreprocessingClassi(AppleStore_data)

#lable for each class
Y=[]
Y=create_classes(target,Y)

Y=np.array(Y)


# get the important features depend on their relationhip
# print(AppleStore_data)
imporfeatModel = ExtraTreesClassifier()
imporfeatModel.fit(AppleStore_data, Y)
#feature_importances of tree based classifiers
important_features=imporfeatModel.feature_importances_
#print("important features are : ",important_features)

#plot graph of feature importances for better visualization
plot_importances = pd.Series(imporfeatModel.feature_importances_, index=AppleStore_data.columns)
plot_importances.nlargest(10).plot(kind='barh')
# plt.show()

joblib.dump(imporfeatModel ,'joblib_imporfeatModel.pkl')


#Y = Y.reshape(AppleStore_data.shape[0],1)
#feature of X Size , Price
#X=pd.DataFrame()
#X['rating_count_ver']=AppleStore_data.iloc[:,3]
#X['ipadSc_urls']=AppleStore_data.iloc[:,7]
#print(AppleStore_data)

X=pd.DataFrame()
X=createXData(X,AppleStore_data)
#print(X.iloc[0,:])
"""
X=pd.DataFrame()
X=AppleStore_data.iloc[:,:4]
X['ver']=AppleStore_data.iloc[:,4]
X['lang.num']=AppleStore_data.iloc[:,12]
#print(X)
"""
X=np.array(X)


#split data into train and test sets
X_train,X_test,Y_train,Y_test =train_test_split(X, Y,test_size=0.20, random_state=0,shuffle=False)

#print(AppleStore_data)
#print(" X_train \n",X_train)
#print(" X_test \n",X_test)

#print(" Y_train \n",Y_train)
#print(" Y_test \n",Y_test)
'''
#get features and target from new test file
newtestdata=pd.read_csv('AppleStore_training_classification.csv')
target_newtest=newtestdata.iloc[:,newtestdata.shape[1]-1]
#print('target new data \n',target_newtest)
newtestdata=createpreprocessingClassi(newtestdata)
Y_newtest=[]
Y_newtest=create_labels(target_newtest,Y_newtest)
Y_newtest=np.array(Y_newtest)
#print('new y',Y_newtest)
X_newtest=pd.DataFrame()
X_newtest=createXData(X_newtest,newtestdata)
X_newtest=np.array(X_newtest)
#print('new X',X_newtest)'''





