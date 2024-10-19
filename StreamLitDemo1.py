import streamlit as st
import pandas as pd
import numpy as np


st.title('Streamlit Demo App')

with st.sidebar:
    st.header('About App')
    st.write('This is a side bar')


st.header('This is a header with divider',divider='rainbow')
st.markdown('This is created using st.markdown')

st.subheader("st.columns")
col1,col2 = st.columns(2)
with col1:
     x=st.slider('Choose an value o f x:',1,10)
with col2:     
    st.write('The value of  :red[***x***] is',x)
    
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

st.area_chart(chart_data)