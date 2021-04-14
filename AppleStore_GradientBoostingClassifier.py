from AppleStore_Milestone2 import*
from sklearn.ensemble import GradientBoostingClassifier
import time
print('\t\t\t Gradient Boosting Classifier Model \t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
start_t=time.time()
GradientBoostingClassifierModel = GradientBoostingClassifier(n_estimators=250, learning_rate=0.01, max_depth=5, random_state=0)
GradientBoostingClassifierModel.fit(X_train, Y_train)
end_t=time.time()

score = GradientBoostingClassifierModel.score(X_train, Y_train)
print("accuracy of training is ",score,'\n trainning time = ',end_t-start_t)

start_t=time.time()
score1 = GradientBoostingClassifierModel.score(X_test, Y_test)
end_t=time.time()
print("accuracy of testing is ",score1,'\n testing time = ',end_t-start_t)
joblib.dump(GradientBoostingClassifierModel,'joblib_GradientBoostingClassifierModel.pkl')




# loaded_model = joblib.load('joblib_GradientBoostingClassifierModel.pkl')
# predict = loaded_model.predict(X_test)
# accuracy = loaded_model.score(X_test, Y_test)
# print('Decission tree accuracy test : ' + str(accuracy),'\n')