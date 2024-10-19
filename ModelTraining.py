
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn 
import numpy as np 
import pickle 
data = pd.read_csv('c:\ML\Algerian_forest_fires_cleaned_dataset.csv')
print(data)

data.drop(['day','month','year'],axis=1,inplace=True)
print(data)

# asigning 0 for not fire and 1 for fire here
data['Classes'] = np.where(data['Classes'].str.contains('not fire'),0,1)

print(data)

print(data['Classes'].value_counts())

print(data.info())

X = data.drop('FWI',axis=1)
y = data['FWI']
print('X values are :\n',X)
print('y values are :\n',y)

#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
print('X Train \n',X_train)
print('X test \n',X_test)

print(X_train.shape)
print(X_test.shape)
#Feature selection based on correlation
print('Correlation :\n',X_train.corr())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score 
model = LinearRegression()

#check for Multicollinearity
plt.figure(figsize=(12,10))
corr = X_train.corr()
sn.heatmap(corr,annot=True) 
plt.show()

def correlation(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
               colname = corr_matrix.columns[i] 
               col_corr.add(colname)
    return col_corr

Corr_features = correlation(X_train,0.85)        
print(Corr_features)

#droping columns correlation greater than 0.85 percentage

X_train.drop(Corr_features,axis=1,inplace=True)   
X_test.drop(Corr_features,axis=1,inplace=True)   

print('X_train shape is:\n' , X_train.shape)
print('X_test shape is:\n' , X_test.shape)

#Feature scaling or standardization 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
#Box plot to understand the effect  of standard scaler
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sn.boxplot(data=X_train)
plt.title('X_train before scaling')
plt.subplot(1,2,2)
sn.boxplot(data=X_train_scaled)
plt.title('X_train After scaling')
plt.show()
 
 
 #Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
LinearReg= LinearRegression()
LinearReg.fit(X_train_scaled,y_train)
y_predict = LinearReg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_predict)
score = r2_score(y_test,y_predict)
print('Mean absolute error :',MAE)
print('Score is :',score)
plt.scatter(y_test,y_predict)
plt.show()

#lasso Regression 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error,r2_score
Lasso_reg= Lasso()
Lasso_reg.fit(X_train_scaled,y_train)
y_predict = Lasso_reg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_predict)
score = r2_score(y_test,y_predict)
print('Lasso Regression Mean absolute error :',MAE)
print('Lasso Regression Score is :',score)
plt.scatter(y_test,y_predict)
plt.show()
 
#Rige Regression 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,r2_score
ridge_reg= Ridge()
ridge_reg.fit(X_train_scaled,y_train)
y_predict = ridge_reg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_predict)
score = r2_score(y_test,y_predict)
print('Ridge Regression Mean absolute error :',MAE)
print('Ridge Regression Score is :',score)
plt.scatter(y_test,y_predict)
plt.show()

#ElasticNet Regression 

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,r2_score
ElasticNet_reg= ElasticNet()
ElasticNet_reg.fit(X_train_scaled,y_train)
y_predict = ElasticNet_reg.predict(X_test_scaled)
MAE = mean_absolute_error(y_test,y_predict)
score = r2_score(y_test,y_predict)
print('ElasticNet Regression Mean absolute error :',MAE)
print('ElasticNet Regression Score is :',score)
plt.scatter(y_test,y_predict)
plt.show()

 
pickle.dump(ElasticNet_reg,open('ElasticNet_reg.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb')) 