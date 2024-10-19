#Polynomial regression 

#importing library 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + 1.5*X  + 2 + np.random.rand(100,1)
# quadratic equation used - y = 0.5x^2+1.5x+2+outliers
plt.scatter(X ,y,color ='g')
plt.xlabel(' X dataset')
plt.ylabel(' Y Dataset')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.20, random_state=10)

from sklearn.linear_model import LinearRegression
Regression_1 = LinearRegression()
Regression_1.fit(X_train,y_train)
from sklearn.metrics import r2_score

score = r2_score(y_test,Regression_1.predict(X_test))
print('Model score =',score)

#lets visualize the model
plt.plot(X_train,Regression_1.predict(X_train),color='r') #this one is for line
plt.scatter(X_train,y_train)
plt.xlabel('X Dataset')
plt.ylabel('Y')
plt.show()
#Lets apply polynomial transformation
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2,include_bias= True)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

regression = LinearRegression()
regression.fit(X_train_poly,y_train)
y_predit = regression.predict(X_test_poly)

score = r2_score(y_test,y_predit)
print('Score using polynomial regression:',score)

plt.scatter(X_train,regression.predict(X_train_poly))
plt.scatter(X_train,y_train)
plt.show()

#prediction of new dataset
X_new = np.linspace(-3,3,200).reshape(200,1)
X_new_poly = poly.transform(X_new)
print(X_new_poly)
print(X_new_poly.shape)

y_new = regression.predict(X_new_poly)
plt.plot(X_new,y_new,"r-",linewidth=2,label='New predictions')
plt.plot(X_train,y_train,'b.',label='Training points')
plt.plot(X_test,y_test,'y.',label='Testing points')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#pipe line 
from sklearn.pipeline import Pipeline

def poly_regression(degree) :    
    X_new = np.linspace(-3,3,200).reshape(200,1)
    ploy_features = PolynomialFeatures(degree=degree,include_bias=True)
    lin_regression = LinearRegression()
    poly_regression = Pipeline([('ploy_features',ploy_features),('lin_reg',lin_regression)])
    #combining two model here 
    poly_regression.fit(X_train,y_train) ## polynomial features and fit using Linear regression
    y_new = poly_regression.predict(X_new)
    print('Inside function this plot is coming')
    plt.plot(X_new,y_new,"r-",linewidth=2,label='New predictions')
    plt.plot(X_train,y_train,'b.',label='Training points')
    plt.plot(X_test,y_test,'y.',label='Testing points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper left')
    plt.show()
    
poly_regression(degree=2)