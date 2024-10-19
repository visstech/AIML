import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression

data = pd.read_csv('c:\\ML\\Hight_Prediction.csv')
print(data)

#Scatter plot 

plt.scatter(x=data['Weight'],y=data['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')
plt.show()

#correlation 
print('Correlation between wight and height\n ',data.corr())

# seaborn for visualization
sns.pairplot(data)
plt.show()

#independent and dependent features
X = data[['Weight']] # Please note independent variable always should DataFrame or 2d array not series. dependent feature may be 1d or 2d , but independent feature shuld alway 2d 
# show we have [[]] here to make it as 2d array or dataframe. 
print(type(X))
print( X.shape)

y = data['Height'] # It may be 1d array are 2d array. 
print(type(y)) # will be series 
print(y.shape)

# Train and test split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=23)

print(X_train.shape)
print(y_train.shape)

# standardization 

scaler = StandardScaler () 
X_train = scaler.fit_transform(X_train) # Note only independent feature should do standardization not for target variable
X_test = scaler.transform(X_test) # only transform not fit_transform -because it should use the mean and standared deviation from Training data set.

# Apply Simple Linear Regression
regression = LinearRegression()

regression.fit(X_train,y_train) # train the model 

print('Coefficient or slope is :',regression.coef_)
print('Interception is:',regression.intercept_)

y_pred = regression.predict(X_test)

print('Predicted values from the test data',y_pred)

#plot training data plot best bit line

plt.scatter(X_train,y_train)
plt.plot(X_train,regression.predict(X_train)) # it will draw the best bit line
plt.show()

# Prediction of test data output = Coefficient + intercept * (Weight)
# y_predict_test =  15.64 +  159.85 * (X_test) THIS IS What HAPPENING 
 
# performance metrics 
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)

print('mean_squared_error is :',mse)
print('mean_absolute_error is :',mae)
print('Root mean_squared_error is :',rmse)

# R square 
# formula 
# R*2 = 1- ssr/sst

from sklearn.metrics import r2_score 

score = r2_score(y_test,y_pred)
print('Our Model score is :',score)

# Adjusted r2 square 

# formula R2 = 1 - [(1-R2) * (n-1)/(n-k-1)]
# R2 the R2 of the model n:Number of observations k:the number of predictor variables
print('Display adjusted r square \n')
adj_r_square = 1 - (  (1-score) * (len(y_test) - 1 ) / (len(y_test) - X_test.shape[1] - 1 ) )
print('Adjusted r square value is:',adj_r_square)

## OLS - Linear Regression model
import statsmodels.api as sm 

model = sm.OLS(y_train,X_train).fit()

prediction = model.predict(X_test)

print(prediction)

print(model.summary())

print('Prediction for the new data')

print('For weight 71 our model prediction is:',regression.predict(scaler.transform([[71]])))
