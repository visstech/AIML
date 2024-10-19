from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st 
import pandas as  pd 
from sklearn.datasets import load_iris

def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data,columns=iris.feature_names)
    df['species'] = iris.target 
    return df,iris.target_names

df,target_names = load_data()
st.sidebar.title('Input Features are as below:')
model = RandomForestClassifier()
model.fit(df.iloc[:,:-1] , df['species'])

sepal_length = st.sidebar.slider('Sepal length',float(df['sepal length (cm)'].min()),float(df['sepal width (cm)'].max()))

sepal_width = st.sidebar.slider('Sepal width',float(df['sepal width (cm)'].min()),float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider('Sepal length',float(df['petal length (cm)'].min()),float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider('Sepal width',float(df['petal width (cm)'].min()),float(df['sepal width (cm)'].max()))


input_data = [[sepal_length,sepal_width,petal_length,petal_width]]
prediction = model.predict(input_data)

species_name = target_names(prediction[0])

st.write('Model Prediction is:',species_name)


