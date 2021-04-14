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




BaggingClassifierModel = BaggingClassifier(base_estimator=SVC(),
                                           n_estimators=90, random_state=0).fit(X_train, Y_train)
score = BaggingClassifierModel.score(X_train, Y_train)
score1 = BaggingClassifierModel.score(X_test, Y_test)
print("accuracy Train 1 is ",score)
print("accuracy test 1 is ",score1)
#90 n_estimators --->>
#accuracy Train is  0.530745179781136
#accuracy test is  0.5104166666666666
'''
#----------------------------------------------------------------------------------------------
# initialize the base classifier
'''


print('\t\t\t BaggingClassifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
base_cls = DecisionTreeClassifier()
# bagging classifier
BaggingClassifierModel2 = BaggingClassifier(base_estimator = base_cls,
                                            n_estimators = 500,
                                            random_state = 20).fit(X_train, Y_train)
score = BaggingClassifierModel2.score(X_train, Y_train)
score1 = BaggingClassifierModel2.score(X_test, Y_test)
print("accuracy Train 2 is ",score)
print("accuracy test 2 is ",score1)
#accuracy Train is  0.9994788952579469
#accuracy test is  0.55625
'''
#----------------------------------------------------------------------------------
'''
BaggingClassifierModel2 = BaggingClassifier()
BaggingClassifierModel2.fit(X_train, Y_train)
score = BaggingClassifierModel2.score(X_train, Y_train)
score1 = BaggingClassifierModel2.score(X_test, Y_test)
print("accuracy Train 3 is ",score)
print("accuracy test 3 is ",score1)
#accuracy Train is  0.9747264200104221
#accuracy test is  0.525

'''
#------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean

model = BaggingClassifier(base_estimator=KNeighborsClassifier())
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=300, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv)
n_scores1 = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv)
print('Accuracy 4: ',mean(n_scores))
print('accurracy 4' , mean(n_scores1))
#Accuracy:  0.5193425925925926
#accurracy1 0.5304166666666666
'''