import streamlit as st 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris


 
 
@st.cache_data
def load_data(): 
   iris = load_iris()
   df = pd.DataFrame(iris.data,columns=iris.feature_names)
   df['species'] = iris.target
   return df,iris.target_names


df,target_name = load_data()

st.sidebar.title('Input Feature Names')

model = RandomForestClassifier()
model.fit(df.iloc[:,:-1] , df['species'])
sepal_length = st.sidebar.slider('Sepal Length',float(df['sepal length (cm)'].min()),float(df['sepal length (cm)'].max()))

sepal_width = st.sidebar.slider('Sepal width',float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))

petal_length = st.sidebar.slider('Petal Length',float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))

petal_width = st.sidebar.slider('Petal width',float(df['petal width (cm)'].min()),float(df['petal width (cm)'].max()))

input_data = [[sepal_length,sepal_width,petal_length,petal_width]]
prediction = model.predict(input_data)
print(prediction)
print(target_name)

predicted_species = target_name[prediction[0]]

st.write('Prediction is:')
st.write(f'Pedicted species name is :{predicted_species}')
