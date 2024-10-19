import streamlit as ST
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


ST.title('My Machine Learning Streamlit App')
ST.info('This is the app developed for machine learning')
with ST.expander('data'):
    ST.write('** data  **')
    df = pd.read_csv('C:\\ML\\2024-08-10T05-49_export.csv')
    df 
    ST.write('** X ** ')
    ST.write('**X**')
    X_raw = df.drop('species', axis=1)
    X_raw

    ST.write('**y**')
    y_raw = df.species
    y_raw   
    
with ST.expander('data Visuvalization' ):
    ST.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
    
# Input features
with ST.sidebar:
  ST.header('Input features')
  island = ST.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'),key=2)
  bill_length_mm = ST.slider('Bill length (mm)', 32.1, 59.6, 43.9,key=11)
  bill_depth_mm = ST.slider('Bill depth (mm)', 13.1, 21.5, 17.2,key=10)
  flipper_length_mm = ST.slider('Flipper length (mm)', 172.0, 231.0, 201.0,key=13)
  body_mass_g = ST.slider('Body mass (g)', 2700.0, 6300.0, 4207.0,key=4)
  gender = ST.selectbox('Gender', ('male', 'female'),key=6)
  

# Input features
with ST.sidebar:
  ST.header('Input features')
  island = ST.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'),key=1)
  bill_length_mm = ST.slider('Bill length (mm)', 32.1, 59.6, 43.9,key=61)
  bill_depth_mm = ST.slider('Bill depth (mm)', 13.1, 21.5, 17.2,key=7)
  flipper_length_mm = ST.slider('Flipper length (mm)', 172.0, 231.0, 201.0,key=8)
  body_mass_g = ST.slider('Body mass (g)', 2700.0, 6300.0, 4207.0,key=9)
  gender = ST.selectbox('Gender', ('male', 'female'),key=3)
  
  # Create a DataFrame for the input features
  data = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)

with ST.expander('Input features'):
  ST.write('**Input penguin**')
  input_df
  ST.write('**Combined penguins data**')
  input_penguins


# Data preparation
# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with ST.expander('Data preparation'):
  ST.write('**Encoded X (input penguin)**')
  input_row
  ST.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={0: 'Adelie',
                                 1: 'Chinstrap',
                                 2: 'Gentoo'})

# Display predicted species
ST.subheader('Predicted Species')
ST.dataframe(df_prediction_proba,
             column_config={
               'Adelie': ST.column_config.ProgressColumn(
                 'Adelie',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Chinstrap': ST.column_config.ProgressColumn(
                 'Chinstrap',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Gentoo': ST.column_config.ProgressColumn(
                 'Gentoo',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
ST.success(str(penguins_species[prediction][0]))  
      