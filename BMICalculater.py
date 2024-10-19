# import the streamlit library
import streamlit as st 

# Define your custom CSS
custom_css = """
<style>
.sidebar .sidebar-content {
    background-color: #f0f2f6;
    color: #262730;
}
</style>
"""
# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Function to change the sidebar color
def change_sidebar_color(color):
    st.markdown(
        f"""
        <style>
        .sidebar .sidebar-content {{
            background-color: {color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
sidebar_color = st.sidebar.color_picker('Choose sidebar color', '#FFFFFF')


 
with st.sidebar:
    change_sidebar_color(sidebar_color)
    
    with st.echo():
         st.write("This code will be printed to the sidebar.")
        
# give a title to our app
st.title('Welcome to BMI Calculator')
 
# TAKE WEIGHT INPUT in kgs
weight = st.number_input("Enter your weight (in kgs)")
 
# TAKE HEIGHT INPUT
# radio button to choose height format
status = st.radio('Select your height format: ',
                  ('cms', 'meters', 'feet'))
 
# compare status value
if(status == 'cms'):
    # take height input in centimeters
    height = st.number_input('Centimeters')
 
    try:
        bmi = weight / ((height/100)**2)
    except:
        st.text("Enter some value of height")
 
elif(status == 'meters'):
    # take height input in meters
    height = st.number_input('Meters')
 
    try:
        bmi = weight / (height ** 2)
    except:
        st.text("Enter some value of height")
 
else:
    # take height input in feet
    height = st.number_input('Feet')
 
    # 1 meter = 3.28
    try:
        bmi = weight / (((height/3.28))**2)
    except:
        st.text("Enter some value of height")
 
# check if the button is pressed or not
if(st.button('Calculate BMI')):
 
    # print the BMI INDEX
    st.text("Your BMI Index is {}.".format(bmi))
 
    # give the interpretation of BMI index
    if(bmi < 16):
        st.error("You are Extremely Underweight")
    elif(bmi >= 16 and bmi < 18.5):
        st.warning("You are Underweight")
    elif(bmi >= 18.5 and bmi < 25):
        st.success("Healthy")
    elif(bmi >= 25 and bmi < 30):
        st.warning("Overweight")
    elif(bmi >= 30):
        st.error("Extremely Overweight")