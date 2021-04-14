from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from  AppleStore_Milestone_cop_1 import*
import sklearn.metrics


print('\t\t\tPolynomial Ridge Regressor Model\t\t\t\n','*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
poly_features = PolynomialFeatures(degree=4)
#X_train_poly = poly_features.fit_transform(X_train)
X_train_poly = poly_features.fit_transform(X_train)

PolynomialRidgeModel = Ridge(alpha=0.1)
PolynomialRidgeModel.fit(X_train_poly, Y_train)
prediction=PolynomialRidgeModel.predict(X_train_poly)
print('Mean Square Error of train = ', metrics.mean_squared_error(Y_train, prediction))
print('accuracy of train = ',metrics.r2_score(Y_train,prediction))

prediction=PolynomialRidgeModel.predict(poly_features.fit_transform(X_test))

print('Mean Square Error of test = ', metrics.mean_squared_error(Y_test, prediction))
print('accuracy of test = ',metrics.r2_score(Y_test,prediction),'\n')
