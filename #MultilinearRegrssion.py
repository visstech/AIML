#Multilinear Regrssion 
from sklearn.datasets import fetch_california_housing
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
import pickle

california = fetch_california_housing()

print(california)
print(california.keys())
print('dependant feature\n',california['target_names'])
print('Independant features\n',california.feature_names)
print('Data \n',california.data)
print('Shape \n',california.data.shape)
print('Shape \n',california.target.shape)

#preparing dataset using DataFrame
dataset = pd.DataFrame(data=california.data,columns=california.feature_names)
print(dataset)
#include the target values also in the dataset 
dataset['Price'] = california.target 
print(dataset)

print(dataset.info())
print(dataset.isnull().sum())
print(dataset.describe())

sns.heatmap(dataset.corr(),annot=True)
plt.show()

X = dataset.iloc[:,:-1] #independant feature
y = dataset.iloc[:,-1] # dependant feature 
print('Independant feature :\n',X.head())
print('y values is\n',y.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=46)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
#Standardscalr only for training and testing dataset only for input feature not for output feature.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Model training 
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
Model = LinReg.fit(X_train,y_train) # Here we are training our model using input feature X_train and its related output feature y_train. y_train contains it output values

print('Coefficient values are :\n',Model.coef_)
print('Intercept values is :\n',Model.intercept_ )

y_predict = Model.predict(X_test) # Prediction using test data

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
score = r2_score(y_test,y_predict)
print('-----------------------------------------------------------------------------------------\n')
print('r2_score of our model is ',score)
print('mean_absolute_error of our model is ',mean_absolute_error(y_test,y_predict))
print('mean_squared_error of our model is ',np.sqrt(mean_absolute_error(y_test,y_predict)))
 
#Adjusted r2 square 

Adjusted_r2_square = 1- ( 1 -score) * ( len(y_test) - 1 ) / ( len(y_test) * X_test.shape[1] - 1 )

print('Adjusted_r2_square value is:\n',Adjusted_r2_square)
print('-----------------------------------------------------------------------------------------')

#Assumption 
plt.scatter(y_test,y_predict)
plt.xlabel('Test truth data')
plt.ylabel('Test Predicted data')
plt.show()

residuals = y_test - y_predict
print('Residuals are :',residuals)
# residuals distributions using displot
sns.displot(residuals,kind='kde')
plt.show()

#scatter plot with prediction with residuals
#uniform distribution
plt.scatter(y_predict,residuals)
plt.show()

print('''
Pickling
Python pickle module is used for serialising and de-serialising a python object structure.
Any object in Python can be pickled so that it can be saved on disk. 
What pickle does is that it "Serialises" the object first before writing it to file. 
Pickle is a way to convert a python object (list,dict,etc.,) into a character stream. 
The idea is that this character stream contains all the information necessary to reconstruct the object in another python script.

''')

#To save our Model
pickle.dump(LinReg,open('LinearRegssionModel.pkl','wb'))
pickle.dump(scaler,open('scaler.pkl','wb')) 
#To load our Model
Loaded_model = pickle.load(open('LinearRegssionModel.pkl','rb'))
 

Newly_predicted = Loaded_model.predict(X_test)
print(Newly_predicted)
