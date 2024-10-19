#Data Encodeing
#1. Nominal/OHE Encoding
#2.Label and ordinal Encoding
#3.Target Guided Ordinal Encoding
#Nominal / OHE Encoding

print(''' One hot encoding also known as nominal Encoding, is a technique used to represent 
categorical data as numerical data, which is more 
suitable for machine learning algorithms. In this techniques,each 
category is represented as binary vector where each bit corresponds to a 
unique category For example, if we have a categorical variable 
"color" with three possible values(red,green,blue), we can 
represent it one hot encoding as follows 

1. Red(1,0,0)
2. Green(0.1,0)
3. Blue(0,0,1)''' )

 
import  pandas as pd; 
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
import numpy as np 
 

## create a single dataframe
df = pd.DataFrame({
    'color':['red','blue','green','green','red','blue']
})

print(df)

encoder = OneHotEncoder()

#Perform fit and transform

encoded = encoder.fit_transform(df[['color']]).toarray() # during fit and transform it will be ordered based on alphabitical values
print(encoded)

encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out())

print(encoded_df)

converted_df = pd.concat([df,encoded_df],axis=1)

print(converted_df)

# One hot encoding with tips data set.
import seaborn as sns 
df = sns.load_dataset('tips')

print(df)

sex_encoded =encoder.fit_transform(df[['sex']]).toarray()
sex_encoded = pd.DataFrame(sex_encoded,columns=encoder.get_feature_names_out())
print(sex_encoded)

encoded_smoker = encoder.fit_transform(df[['smoker']]).toarray()
encoded_smoker = pd.DataFrame(encoded_smoker,columns=encoder.get_feature_names_out())
print(encoded_smoker)

encoded_day = encoder.fit_transform(df[['day']]).toarray()
encoded_day = pd.DataFrame(encoded_day,columns=encoder.get_feature_names_out())
print(encoded_day)

encoded_time = encoder.fit_transform(df[['day']]).toarray()
encoded_time = pd.DataFrame(encoded_time,columns=encoder.get_feature_names_out())
print(encoded_time)

df1 = pd.concat([df,sex_encoded],axis=1)
print(df1)
df2 = pd.concat([df1,encoded_smoker],axis=1)

df3 = pd.concat([df2,encoded_day],axis=1)

final_data = pd.concat([df3,encoded_time],axis=1)
print(final_data)

final_data.drop(['sex','smoker','day','time'],axis=1,inplace=True)
print(final_data)


print (''' #Label Encoding
#Label encoding involves assigning a unique numerical label to each category in the variable. 
The labels are usually assigned in alphabetical order or based on 
the frequency of categories. For example, if we have a 
categorical variable "color" with three possible values(red,green,blue), we 
can represent it using label encoding as follows: 
    
    1.Red 1
    2. Green 2
    3. Blue 3
    
''')

## create a single dataframe
df = pd.DataFrame({
    'color':['red','blue','green','green','red','blue']
})

Lencoder = LabelEncoder()
Lbl_encoded = Lencoder.fit_transform(df[['color']]) 
print(Lbl_encoded)


print(Lencoder.fit_transform([['blue']]))

print('''  
                Ordinal Encoding
                It is used to encode categorical data that have an intrinsic order or ranking. 
                In this technique, each category is assigned a numerical value based on its position
                in the order. For example, if we have a categorical variable "education level" with four
                possible values (high school,collage,graduate,post-graduate), we can represent it using ordinal
                encoding as follows
                1. High school : 1
                2. College : 2
                3. Graduate : 3
                4. Post-Graduate: 4


''')

#Example
size_df = pd.DataFrame({'Size':['large','medium','small','large','small']})
print(size_df)

from sklearn.preprocessing import OrdinalEncoder

Ord_encoder = OrdinalEncoder(categories=[['small','medium','large']])

Oencoded = Ord_encoder.fit_transform(size_df[['Size']])

print(Oencoded)

print(
            ''' 
            3. Target Guided Ordinal Encoding
            It is a technique used to encode categorical variables based on their ralationship with the target variable.
            This encoding technique is useful when we have a categorical variable with a large number of unique categories, and we want to 
            use this variable as feature in our machine learning model.
            
            In Target Guided Ordinal Encoding. We replace each category in the categorical variable with a numerical value based on the mean or
            median of the target variable for that category. This creates a monotonic relationship between the categorical variable and the target variable,
            which can improve the predictive power of our model.
            
            ''')

#Example:
city_df = pd.DataFrame({'city':['New York','Paris','Tokyo','New York','Paris'],
                        'price':[200,100,150,260,170]})

print(city_df)

city_mean = city_df.groupby('city')['price'].mean()

print('city mean values :',city_mean)
 
city_df['city_encoded'] = city_df['city'].map(city_mean)

print(city_df)
 
 # Another example using tips data set to change the time categorical variable as numerical value using Target Guided Encoding
 
TGE_df = sns.load_dataset('tips')
print(TGE_df)
 
time_mean = TGE_df.groupby('time')['total_bill'].mean()
print(time_mean)
 
TGE_df['time_encoded'] = TGE_df['time'].map(time_mean) 
print(TGE_df)
 