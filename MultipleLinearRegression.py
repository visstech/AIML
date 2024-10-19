# https://github.com/krishnaik06 
#Multiple linear regression example:

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from sklearn.model_selection import train_test_split

data = pd.read_csv('c:\\ML\\economic_index.csv')
print(data)
print(data.columns)
# drop unnecessary columns 

data.drop(['Unnamed: 0','year','month'],axis=1,inplace=True)

print(data)

print('Check the null values count\n',data.isnull().sum())

#checking pairplot to check the correlation 

sns.pairplot(data)
plt.show()

print(data.corr()) # Here it we check interest_rate with index_price having positive correlation( 0.935793) and interest_rate with unemployment_rate 
                   # having negative correlation( -0.925814)

# visualize the data points more closely 

plt.scatter(data['interest_rate'],data['unemployment_rate']) 
plt.xlabel('interest_rate')
plt.ylabel('unemployment_rate')
plt.show()

plt.scatter(data['interest_rate'],data['index_price']) 
plt.xlabel('interest_rate')
plt.ylabel('index_price')
plt.show()

#independent feature and dependent feature
X = data[['interest_rate','unemployment_rate']]
print(X.shape)
y = data['index_price']
print(y.shape)

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)

sns.regplot( x='interest_rate' , y='index_price' ,data=data) 
plt.xlabel('interest_rate')
plt.ylabel('index_price')
plt.show()
 
sns.regplot( x='interest_rate' , y='unemployment_rate' ,data=data) 
plt.xlabel('interest_rate')
plt.ylabel('unemployment_rate')
plt.show()

#standardization 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 
Regression  = LinearRegression ()
Regression.fit(X_train,y_train)
 

# Cross validation 
cross_validation = cross_val_score(Regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3) 

print(cross_validation)
print('cross validation mean :',np.mean(cross_validation))

y_predict = Regression.predict(X_test)

print(y_predict) 

# performance metrics 
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse = mean_squared_error(y_test,y_predict)
mae = mean_absolute_error(y_test,y_predict)
rmse = np.sqrt(mse)

print('mean_squared_error is :',mse)
print('mean_absolute_error is :',mae)
print('Root mean_squared_error is :',rmse)

# R square 
# formula 
# R*2 = 1- ssr/sst

from sklearn.metrics import r2_score 

score = r2_score(y_test,y_predict)
print('Our Model score is :',score)

# Adjusted r2 square 

# formula R2 = 1 - [(1-R2) * (n-1)/(n-k-1)]
# R2 the R2 of the model n:Number of observations k:the number of predictor variables
print('Display adjusted r square \n')
adj_r_square = 1 - (  (1-score) * (len(y_test) - 1 ) / (len(y_test) - X_test.shape[1] - 1 ) )
print('Adjusted r square value is:',adj_r_square)

#Assumption 
plt.scatter(y_test,y_predict)
plt.show()

residuals = y_test - y_predict # error
print('Residuals :',residuals)

# plot this residuals 
sns.displot(residuals,kind='kde')
plt.show()

plt.scatter(y_predict,residuals)
plt.xlabel('Predicted')
plt.ylabel('Error')
plt.show()

## OLS - Linear Regression model
import statsmodels.api as sm 

model = sm.OLS(y_train,X_train).fit()

prediction = model.predict(X_test)

print(prediction)

print(model.summary())
print('Regression model Coefficient value is\n')
print(Regression.coef_)

new_value_prediction = Regression.predict(scaler.transform([[4.6,7.5]]))
print('new_value_prediction:',new_value_prediction)