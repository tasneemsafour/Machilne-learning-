import numpy as np
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
import joblib

def drop_row(data):
    data.dropna(how='any', inplace=True)
    return data

def drop_columns(data,cols):
 for col in cols:
    data.drop(col, axis=1, inplace=True)
 return data


def OneHot_Encoder(data,col):

    hot_encoderModel = ce.OneHotEncoder(cols=col,use_cat_names =True )
    hot_encoderModel.fit(data)
    data =hot_encoderModel.transform(data)
    f=('joblib_hot_encoderModel.pkl')
    joblib.dump(hot_encoderModel,f)
    return data



def label_encoder(data, cols):
    dict_all = dict(zip([], []))
    label_encoderModel = LabelEncoder()
    for col in cols:
        temp_keys = data[col].values
        temp_values = label_encoderModel.fit_transform(data[col])
        dict_temp = dict(zip(temp_keys, temp_values))
        dict_all[col] = dict_temp
        data.replace(dict_all[col], inplace=True)
    joblib.dump(dict_all, 'joblib_label_encoderModel.pkl')
    return data

def featureScaling(data):
    Normalized_data = np.zeros((data.shape[0], data.shape[1]));
    for i in range(data.shape[1]):
        Normalized_data[:, i] = (data[:, i] - min(data[:, i])) / (max(data[:, i]) - min(data[:, i]));
    return Normalized_data









