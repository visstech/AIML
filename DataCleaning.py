import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn 
import numpy as np 

data = pd.read_csv('C:\ML\Algerian_forest_fires_dataset_UPDATE.csv',header=1)
print(data)
print(data.info()) # All are object data type
#data cleaning 
print(data.isnull().sum()) # checking missing values
data[data.isnull().any(axis=1)]
print(data[data.isnull().any(axis=1)])

data.loc[:122,'Region'] =  0 
data.loc[122:,'Region'] =  1 
df = data 
print(df.info())

df['Region'] = df['Region'].astype(int)

print(df.info())

df = df.dropna().reset_index(drop=True)

print(df.isnull().sum())
print(df.iloc[[122]])

# removing 122nd row
df = df.drop(122).reset_index(drop=True)
print(df.iloc[[122]])
print(df.columns)
#remove space in the columns name
df.columns = df.columns.str.strip()
print(df.columns)

#changing the required columns as integer data type
df[['day','month','year','Temperature','RH','Ws']] = df[['day','month','year','Temperature','RH','Ws']].astype(int)
print(df.info())

#df[[ 'Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']] = df[[ 'Rain', 'FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)
#selecting only the object data type columns in the objects list 
objects = [features for features in df.columns if df[features].dtypes=='O']
print('data type',objects)

for i in objects:
    if i != 'Classes': # changing the columns as float other then Classes column
       df[i] = df[i].astype(float)

print(df.info())

print(df.describe())

#Let save the cleaned dataset
df.to_csv('C:\ML\Algerian_forest_fires_cleaned_dataset.csv',index=False)
#Exploratory data analysis:
df_copy = df.drop(['day','month','year'],axis=1)
#encoding the classes column to numberic value
#df_copy['Classes'] = np.where(df_copy['Classes']=='not fire',0,1)

print(df_copy)
print(df_copy['Classes'].value_counts())
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'),0,1)
print(df_copy)
print(df_copy['Classes'].value_counts())

#PLOT Density plot for all the features
#plt.style.use('seaborn-paper')
df_copy.hist(bins=50,figsize=(20,15))
plt.show()

#Percentage for pie chart
percentage = df_copy['Classes'].value_counts(normalize=True) * 100 # it will provide the percentage of fire and not fire
#ploting chart
classLabels = ['fire','not fire']
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classLabels,autopct='%1.1f%%')
plt.show()

#checking correlation between all features
print(df_copy.corr())

sn.heatmap(df_copy.corr())
plt.show()

sn.boxplot(df_copy['FWI'],color='Green')
plt.show()

df['Classes'] = np.where(df['Classes'].str.contains('not fire'),'not fire','fire')

#Monthly fire analysis 
dftemp = df.loc[df['Region'] == 1] 
plt.subplots(figsize=(13,6))
sn.set_style('whitegrid')
sn.countplot(x ='month',hue='Classes',data=df)
plt.ylabel('Number of Fire',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title('Fire Analysis of Sidi-Bel Abbes',weight='bold')
plt.show()


dftemp = df.loc[df['Region'] == 0] 
plt.subplots(figsize=(13,6))
sn.set_style('whitegrid')
sn.countplot(x ='month',hue='Classes',data=df)
plt.ylabel('Number of Fire',weight='bold')
plt.xlabel('Months',weight='bold')
plt.title('Fire Analysis of  Bejaia Region',weight='bold')
plt.show()