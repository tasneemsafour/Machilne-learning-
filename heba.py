from sklearn.linear_model import LogisticRegression
import numpy as np
from AppleStore_Milestone2 import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
#-----------------------ACCuracy = 49.--------------------------------------------
'''
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
#Generate a confusion matrix
score = model.score(X_test, Y_test)
print("accuracy is ",score)
'''
#-----------------------ACCuracy = 49.--------------------------------------------
'''
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
score = model.score(X_test, Y_test)
print("accuracy is ",score)
# Compute and print the confusion matrix and classification report
#print(confusion_matrix(Y_test, Y_predict))
'''
#---------------------------------------Accuracy = 68-----------------------------------------
'''
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, Y_train)
score = model.score(X_train, Y_train)
score1 = model.score(X_test, Y_test)
print("accuracy is ",score,score1)

'''
'''
#-----------------------------------96 but 45--------------------------------------------------------
m = VotingClassifier(
    estimators=[('lr', LogisticRegression()),
                ('rf', RandomForestClassifier()),
                ('gnb', GaussianNB())],
    voting='hard')
m.fit(X_train, Y_train)
print(m.score(X_train, Y_train))
print(m.score(X_test, Y_test))
'''
#-----------------------------------------------------------------------------------------------------
'''
clf = BaggingClassifier(base_estimator=SVC(),
                        n_estimators=90, random_state=0).fit(X_train, Y_train)
score = clf.score(X_train, Y_train)
score1 = clf.score(X_test, Y_test)
print("accuracy Train is ",score)
print("accuracy test is ",score1)
#90 n_estimators --->>
#accuracy Train is  0.530745179781136
#accuracy test is  0.5104166666666666
'''
#----------------------------------------------------------------------------------------------
# initialize the base classifier
'''
base_cls = DecisionTreeClassifier()
# bagging classifier
model = BaggingClassifier(base_estimator = base_cls,
                          n_estimators = 500,
                          random_state = 20).fit(X_train, Y_train)
score = model.score(X_train, Y_train)
score1 = model.score(X_test, Y_test)
print("accuracy Train is ",score)
print("accuracy test is ",score1)
#accuracy Train is  0.9994788952579469
#accuracy test is  0.55625
'''
#----------------------------------------------------------------------------------
'''
model = BaggingClassifier()
model.fit(X_train, Y_train)
score = model.score(X_train, Y_train)
score1 = model.score(X_test, Y_test)
print("accuracy Train is ",score)
print("accuracy test is ",score1)
#accuracy Train is  0.9747264200104221
#accuracy test is  0.525

'''
#------------------------------------------------------------------------------------
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean

model = BaggingClassifier(base_estimator=KNeighborsClassifier())
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=400, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv)
n_scores1 = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv)
print('Accuracy: ',mean(n_scores))
print('accurracy1' , mean(n_scores1))
#Accuracy:  0.5193425925925926
#accurracy1 0.5304166666666666
'''
#----------------------------------------------------------------------------------------------------------------
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline =  LogisticRegression()
pipeline.fit(X_train, Y_train)
score = pipeline.score(X_train, Y_train)
score1 = pipeline.score(X_test, Y_test)
print("accuracy Train is ",score)
print("accuracy test is ",score1)
#--------------------------------------------------------------------------------------------
bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100, random_state=1, n_jobs=5)
pipeline.fit(X_train, Y_train)
score = pipeline.score(X_train, Y_train)
score1 = pipeline.score(X_test, Y_test)
print("accuracy Train is ",score)
print("accuracy test is ",score1)