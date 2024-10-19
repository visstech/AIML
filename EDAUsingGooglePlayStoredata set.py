#EDA Using google play store data set
#Steps we are going to follow
# 1. Data cleaning 
# 2. Exploratory data analysis
# 3. Feature Engineering

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from  sklearn.preprocessing import OneHotEncoder, LabelEncoder,OrdinalEncoder
import datetime 

Data = pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')
print(Data)

print(Data.info())
print('Check the Nan values count')
print(Data.isnull().sum())

# To get the summary details
print(Data.describe()) # Since we have only one numerical feature Rating only shown here.
print('Data cleaning techniques ')
#checking Reviews columns
print('Checking unique values for Review column:',Data['Reviews'].unique())
#Checking ghe numeric value count for review columns

print(Data['Reviews'].str.isnumeric().sum())

print('To get the not numeric values records')

print(Data[~Data['Reviews'].str.isnumeric()]) # if you see for review 3.0M is available as non numeric value.
# Just coping the original data set into Data_copy Dataframe and cleaning data using this data frame.

Data_copy = Data.copy()
#Removing Non numeric record from the data set
Data_copy.drop(Data_copy.index[10472],inplace=True)
print('After dropping the Non numeric record for Reviews columns:')
print(Data_copy[~Data_copy['Reviews'].str.isnumeric()]) # it will not show anything because it is dropped already

print(' Now convert Reviews column into int datatype')

Data_copy['Reviews'] = Data_copy['Reviews'].astype(int)

print('Data cleaning on Size colum:')
print(Data_copy['Size'].unique()) #Here M for million and K for thousand is there we need to remove this to make it as numeric value

Data_copy['Size'] = Data_copy['Size'].str.replace('M','000')
Data_copy['Size'] = Data_copy['Size'].str.replace('k','')
print(Data_copy['Size'].unique())  #After replaced 
print('Checking Non value in Size Column',Data_copy['Size'].isnull().sum())
Data_copy['Size'] = Data_copy['Size'].str.replace('Varies with device','0') #Non numeric replaced with 0
print(Data_copy['Size'].unique())

Data_copy['Size'] = Data_copy['Size'].astype(float) #changing the data type from object to float.

print('Data cleaning for columns Installs,Price')
print(Data_copy['Price'].unique())
print(Data_copy['Installs'].unique())

character_to_remove =['+',',','$']
columns_to_clean = ['Installs','Price']

for item in character_to_remove:
    for cols in columns_to_clean:
       Data_copy[cols] = Data_copy[cols].str.replace(item,'')
       
       
print(Data_copy['Price'].unique())
print(Data_copy['Installs'].unique())
#Changing the data type from object tyoe to int type 
Data_copy['Price'] = Data_copy['Price'].astype('float')
Data_copy['Installs'] = Data_copy['Installs'].astype('int')

print(Data_copy.info())

#Handling last update columns 
print(Data_copy['Last Updated'].unique())
Data_copy['Last Updated'] = pd.to_datetime(Data_copy['Last Updated'])
print(Data_copy['Last Updated'].unique())

Data_copy['Day'] = Data_copy['Last Updated'].dt.day 
Data_copy['Month'] = Data_copy['Last Updated'].dt.month
Data_copy['Year'] = Data_copy['Last Updated'].dt.year 

print(Data_copy.info())

Data_copy.to_csv('c:\\ML\\GoogleStore_Data_cleaned.csv')
#Checking the categorical feature App Column
print(Data_copy['App'].unique())
print(Data_copy['App'].isnull().sum())

#EDA Start here.
print('Checking the duplicate values in App columns:')
print(Data_copy[Data_copy['App'].duplicated()])
print('Shape of duplicated records in the dataset is:')
print(Data_copy[Data_copy['App'].duplicated()].shape)
print('Observation is that App column has duplicate values')

Data_copy = Data_copy.drop_duplicates(subset='App',keep='first')

print('After deleting duplicate records in App columns',Data_copy.shape)

print('Explore the data')

Numerical_features = [feature for feature in Data_copy.columns if Data_copy[feature].dtype != 'O' ]
Categorical_features = [feature for feature in Data_copy.columns if Data_copy[feature].dtype == 'O' ]
print(f'We have {len(Numerical_features)} features in our dataset {Numerical_features}' )
print(f'We have {len(Categorical_features)} features in our dataset {Categorical_features}' )

print('proportion  of count data on categorical columns')

for cols in Categorical_features:
    print(Data_copy[cols].value_counts(normalize=True) * 100)
    print('----------------------------------------')
    
print('proportion  of count data on numerical columns')

plt.figure(figsize=(15,15))
plt.suptitle('Univariate analysis on numerical features',fontsize = 20,fontweight='bold',alpha=0.8,y=1)

for i in range(0,len(Numerical_features)):
    plt.subplot(5,3, i+1)
    sns.kdeplot(x=Data_copy[Numerical_features[i]],shade='True',color='g')
    plt.xlabel(Numerical_features[i])
    plt.tight_layout()
    
plt.show()

print('Observation Rating and year are left skewed while Reviews ,price,installs and sise are right skewed')

plt.figure(figsize=(15,15))
plt.suptitle('Univariate analysis on categorical features',fontsize = 20,fontweight='bold',alpha=0.8,y=1)
category = ['Type','Content Rating']
for i in range(0,len(category)):
    plt.subplot(2,2, i+1)
    sns.countplot(x=Data_copy[category[i]],palette='Set2')
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
plt.show()
print('Which is the most popular App Category:')
Data_copy['Category'].value_counts().plot.pie(y=Data_copy['Category'],figsize=(15,16),autopct='%1.1f') # autopct='%1.1f' to get the percentage values
plt.show()

#  Top ten App Category 

Category = pd.DataFrame(Data_copy['Category'].value_counts())
print(Category)
Category.rename(columns={'Category':'count'},inplace=True)

print(Category)

print('Bar chart of Top ten App Categories')
plt.figure(figsize=(15,6))
sns.barplot(x= Category.index[:10],y='count',data = Category[:10],palette='hls')
plt.title('To ten App Categories')
plt.xticks(rotation = 45)
plt.show()

#Which category has largest number of installations?

Installation = pd.DataFrame(Data_copy[['Category','Installs']])

print('Which category has largest number of installations:\n',Installation.groupby('Category').sum('Installs').sort_values(by='Installs',ascending=False)[:1])
print('What are the to 5 most installed App in Each category?:\n', Installation.groupby('Category').sum('Installs').sort_values(by='Installs',ascending=False)[:5])
print('How many App are there in google play store which get top 5 rating:\n',len(Data_copy[Data_copy['Rating'] == 5]))
