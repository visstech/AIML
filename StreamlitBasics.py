import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 

st.title(' This is streamlit basic app')

# 2. Header and Subheader:

st.header('This is a sample header')

st.subheader('This is sample sub-header')

#text
st.text('This is documentation about streamlit basic commands')

#Markdown
st.markdown('# Markdown with simple #charactor')
st.markdown('## This is markdown sample with ## charactor ')
st.markdown('### This is markdown sample with ### charactor ')


# Success, Info, Warning, Error, Exception:

st.success('Success')

st.info('Information')

st.warning('Warning text to display')

st.error('The error text to display')
exp = ZeroDivisionError("Trying to divide by Zero")
st.exception(exp)

st.text(' Write:\n'

'Using write function, we can also display code in coding format. This is not possible using st.text(”).')

# Write text
st.write("Text with write")
 
# Writing python inbuilt function range()
st.write(range(10))
for i in range(1,50):
    st.write(i,'*', i ,'= ', i * i )

# Display Images
 
# import Image from pillow to open images

from PIL import Image
img = Image.open("C:\ML\KRUTHIKHA.jpg")
st.text('My DAUGHTER kRUTHISHASHINI')
st.image(img, width=400)
st.balloons()

st.text('Checkbox:\n'

'A checkbox returns a boolean value. When the box is checked, it returns a True value else returns a False value.')

st.checkbox('Show Balloons/Hide balloons')

if st.checkbox('sHOW / hIDE'):
    st.write('You check the check box')
else:
    st.write('You have unchecked the check box')
    

# radio button
# first argument is the title of the radio button
# second argument is the options for the radio button

status = st.radio('Select the Gender :',('Male','Female'))

if status == 'Male':
    st.success('Male has been selected.')
else:
    st.success('Female has been selected')
    
    
st.text('Selection Box:\n'

'You can select any one option from the select box')


hobby  = st.selectbox('Please select your hobby ',('Dancing','singing','playing','summing'))

st.write(hobby)


st.text('Multi-Selectbox:\n'

'The multi-select box returns the output in the form of a list. You can select multiple options.')

Myhobbies = st.multiselect('Please select you hobbies',['Dancing','Reading','Running','Playing','music'])

st.write("You selected", len(Myhobbies), Myhobbies)

st.text('Button:\n'

'st.button() returns a boolean value. It returns a True value when clicked else returns False.')

st.button('Submit')

if(st.button("About")):
    st.text("Welcome To Learning basics of streamlit app!!!")
    
# Text Input
 
# save the input text in the variable 'name'
# first argument shows the title of the text input box
# second argument displays a default text inside the text input area

name = st.text_input('Please enter your name','Type your name here....')

if st.button('showname'):
  result = name.title()
  st.success(result)
  
# slider
 
# first argument takes the title of the slider
# second argument takes the starting of the slider
# last argument takes the end number

level = st.slider('select the range of values',1,20)
st.text('selected {}'.format(level))


# Adding a slider to the sidebar
age = st.sidebar.slider('Your age', 0, 130, 25)

# Adding a selectbox
favorite_color = st.sidebar.selectbox('Favorite color', ['Blue', 'Red', 'Green'])

st.text('Expander:\n'
'Use st.expander to hide and reveal detailed information, making your app cleaner and easier to navigate.')
with st.expander('See details'):
    st.write('Detailed information here.')
st.write('Columns example:')
col1, col2 = st.columns(2)
col1.write('Column 1 Content')
col2.write('Column 2 Content')

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    "## This is the sidebar"
    
st.write('Input widgets\n'
'With widgets, Streamlit allows you to bake interactivity directly into your apps with buttons, sliders, text inputs, and more.')
    
import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live

st.write("# Streamlit camera input live Demo\n"
"## Try holding a qr code in front of your webcam")

image = camera_input_live()

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    detector = cv2.QRCodeDetector()

    data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    if data:
        st.write("# Found QR code")
        st.write(data)
        with st.expander("Show details"):
            st.write("BBox:", bbox)
            st.write("Straight QR code:", straight_qrcode)


#Streamlit camera input example

picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)

st.write('Display a toggle widget')
st.write('syntax: st.toggle(label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")')

on = st.toggle("Activate feature")

if on:
    st.write("Feature activated!")

st.markdown('#st.link_button')

st.write('Display a link button element.\n'

'When clicked, a new tab will be opened to the specified URL. This will create a new session for the user if directed within the app.')

st.write('st.link_button(label, url, *, help=None, type="secondary", disabled=False, use_container_width=False)')

st.link_button("Go to gallery", "https://streamlit.io/gallery")

st.markdown('# st.feedback')
st.write('A feedback widget is an icon-based button group available in three styles, as described in options. \nIt is commonly used in chat and AI apps to allow users to rate responses.')

sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")