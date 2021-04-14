from AppleStore_Milestone2 import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
import time
print('\t\t\t Extra Tree Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

start_t=time.time()
ExtraTreeClassifierModel = ExtraTreeClassifier(random_state=0,max_depth=12)
ExtraTreeClassifierModel=BaggingClassifier(ExtraTreeClassifierModel, random_state=0).fit(X_train, Y_train)
end_t=time.time()

max_d=12
print('max depth = ',max_d,'\n accuracy of training is : ',ExtraTreeClassifierModel.score(X_train, Y_train),'\n trainning time = ',end_t-start_t)

start_t=time.time()
acc=ExtraTreeClassifierModel.score(X_test, Y_test)
end_t=time.time()

print('accuracy of testing is : ',acc,'\n testing time = ',end_t-start_t)
joblib.dump(ExtraTreeClassifierModel,'joblib_ExtraTreeClassifierModel.pkl')



# loaded_model = joblib.load('joblib_ExtraTreeClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')