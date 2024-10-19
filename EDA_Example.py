#EDA 
import pandas as pd 
data = pd.read_csv('C:\\ML\\winequality-red.csv') 
print(data)

print(data.info())
# Descriptive summary of the dataset
print(data.describe())

# To check the number of columns and rows in the dataset
print(data.shape)

# To print down all the columns in the data set

print(data.columns)

print('Target variable in wine data set is quality:',data['quality'].unique())

# To check the missing values in the data set
print(data.isnull().sum())

# To check the duplicated values in the data set
print(data.duplicated())

# only display the duplicated records 
print('To display the duplicated records only by using below code:')
print(data[data.duplicated()])

print('To remove the duplicated records use below code :')
print(data.drop_duplicates(inplace=True))
print(data.shape)

print('To see the correlation between each features:')
print(data.corr())

print('To view in graphical manner using seaborn:')
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(),annot=True)
plt.show()

#visualization 
# It is a imbalanced data 
plt.xlabel('Wine Quality')
plt.ylabel('Count')
print(data.quality.value_counts().plot(kind='bar'))
plt.show()

for column in data.columns:
    sns.histplot(data[column],kde=True)
plt.show()

# to see the each colum separetly 

sns.histplot(data['fixed acidity'],kde=True)
plt.show()

#Univariate,Bivariate,multivariate analysis
sns.pairplot(data)
plt.show()

#Categorical plot
sns.catplot(x='quality',y='alcohol',data = data , kind='box')
plt.show()

sns.scatterplot(x='alcohol',y='pH',hue='quality',data=data)
plt.show()