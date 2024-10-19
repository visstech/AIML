#EDA using flight price dataset
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_excel('C:\\ML\\flight_price.xlsx')
print(data)

# to get the basic data set info

print(data.info()) 

# to get the statistical information
print(data.describe()) # since we have only one numerical columns it shows only one column

#Feature Engineering start from here
data['date'] = data['Date_of_Journey'].str.split('/').str[0]
print(data['date'])
data['Month'] = data['Date_of_Journey'].str.split('/').str[1]
data['Year'] = data['Date_of_Journey'].str.split('/').str[2]
print(data['Year'])
print(data.info())
#Changing the data type of above columns from object to int

data['date'] = data['date'].astype(int) 
data['Month'] = data['Month'].astype(int)
data['Year'] = data['Year'].astype(int)

print(data.info())  # if we check the the data type will be changed now

# dropping the Date_of_Journey column now it is no more required
data.drop('Date_of_Journey',axis=1,inplace=True)

data['Arrival_Time'] = data['Arrival_Time'].apply(lambda x:x.split( )[0]) # split based on empty space because arival time has 10:10 22 mar like this we 
#want only the hours and minuit only 
data['Arrival_Hours'] = data['Arrival_Time'].str.split(':').str[0]
data['Arrival_Min'] = data['Arrival_Time'].str.split(':').str[1]

print(data.info())
#changing the data type of newly created columns
data['Arrival_Hours'] = data['Arrival_Hours'].astype(int)
data['Arrival_Min'] = data['Arrival_Min'].astype(int)


print(data.info())

# Drop the column Arrival_Time now it is no more required 
data.drop('Arrival_Time',axis=1,inplace=True)

# Dep_Time feature engineering 
data['Dep_Hours'] = data['Dep_Time'].str.split(':').str[0]
data['Dep_Min'] = data['Dep_Time'].str.split(':').str[1]

data['Dep_Hours'] = data['Dep_Hours'].astype(int)
data['Dep_Min'] = data['Dep_Min'].astype(int)
#Dropping departure time column now it no more required 
data.drop('Dep_Time',axis=1,inplace=True)

print(data.info())
# Handling Total_Stops columns
print(data['Total_Stops'].unique())
data['Total_Stops'] = data['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,np.nan:1})
print('Nan value count for Total_Stops:')
print(data['Total_Stops'].isnull().sum())

# Dropping Route column
data.drop('Route',axis=1,inplace=True)
print('Handling Duration column values')
print(data['Duration'])
data['Duration_hours'] = data['Duration'].str.split(' ' ).str[0].str.split('h').str[0]

data['Duration_Min'] = data['Duration'].str.split(' ' ).str[1].str.split('m').str[0]
data['Duration_Min'] =  data['Duration_Min'].fillna('0')
data['Duration_Min'] = data['Duration_Min'].astype(int)

print(data['Duration_Min'].unique())
data.drop('Duration',axis=1,inplace=True)

print(data)

# Now handling categorical variable using one hot encoding 

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

encoded_data = encoder.fit_transform(data[['Airline','Source','Destination','Additional_Info']]).toarray()
print(encoded_data)

encoded_df = pd.DataFrame(encoder.fit_transform(data[['Airline','Source','Destination','Additional_Info']]).toarray(),columns=encoder.get_feature_names_out())

print(encoded_df)

# concatenate the encoded_df with original data frame
final_df = pd.concat([data,encoded_df],axis=1)

final_df.drop(['Airline','Source','Destination','Additional_Info'],axis=1,inplace=True)
print(final_df)
print(final_df.columns)