import streamlit as st 


st.title(' Welcome to python practice quiz')
st.write('1. Which method used to update existing values in Set in python?')
option = st.radio(
    "Choose the answer",
     ("Add", "Update", "Append","extend"),
      index=None  # This ensures that nothing is selected initially)
     )

       
if 'checkbox_value' not in st.session_state:
    st.session_state.checkbox_value = False

opt1correct =   """ ### Code for example \n
set1 = {2,3,5,6}\n
set2 = {7,3,9,6}\n
              
update =  {7:27,9:29}  
              
    for oldvalue,newvalue in update.items(): 
        set2.discard(oldvalue) 
        set2.add(newvalue) 
print('Updated set value now is:',set2)
              
        """
if option == "Add" :
    st.write(':green[Excellent Correct Answer!!!!!!!!!]')
    st.write(opt1correct)
    
if option != "Add" and option is not None  :
    st.write(':red[Sorry, Wrong Answer]')
    st.write(opt1correct)
#checked_option = []
#option1 = st.checkbox('1.add',value=st.session_state.checkbox_value)
#option2 =st.checkbox('2.update',value=st.session_state.checkbox_value)
#option3 =st.checkbox('3.append',value=st.session_state.checkbox_value)
#option4 =st.checkbox('4.extend',value=st.session_state.checkbox_value)
 
#if option1:
 #   checked_option.append(option1)
#elif  option2:
 #   checked_option.append(option2)
    
#if option1:
#    st.write('Excellent Correct Answer!!!!!')
#if  option2 :
 #   st.write('Wrong answer')
 #   st.write(opt1correct)
#if  option3:
#    st.write('Wrong answer')
#    st.write(opt1correct)
#if option4:
#    st.write('Wrong answer')
#    st.write(opt1correct)

st.write('2. How do you create a set in Python?')
option = st.radio(
    "Choose the answer",
     ("using {} or set() function", "using bracket []", "using set=[{}]"),
      index=None  # This ensures that nothing is selected initially)
     )
correct =""" Code Example :

### Using curly braces
my_set = {1, 2, 3}

### Using the set() function
my_set = set([1, 2, 3])
""" 
if option == "using {} or set() function" :
    st.write(':green[Excellent Correct Answer!!!!!!!!!]')
    
if option != "using {} or set() function" and option is not None  :
    st.write(':red[Sorry, Wrong Answer]')
    st.write(correct)

st.write('3. Which of the following is a property of Python sets?')
option = st.radio(
    "Choose the answer",
     (  "A) Sets are ordered collections of elements",
        "B) Sets allow duplicate elements",
        "C) Sets are mutable",
        "D) Sets support indexing"),
      index=None  # This ensures that nothing is selected initially)
     )

if option == "C) Sets are mutable" :
    st.write(':green[Excellent Correct Answer: C) Sets are mutable]')
    
if option != "C) Sets are mutable" and option is not None :    
     
    st.markdown(""" 
             :red[Sorry Wrong answer]\n
             :green[Correct answer is :  C) Sets are mutable.] 
             ## Explanation:
             Python sets are mutable, allowing you to add or remove elements, but they do not support duplicates, 
             indexing, or ordering.
             
             """)
    
st.write("4. How do you create an empty set in Python?")
option = st.radio(
    "Choose the answer",
    ( "A) s = {}",
      "B) s = set()",
      "C) s = []",
      "D) s = () "),
    index=None
)

if option == "B) s = set()" :
   st.write(':green[Excellent correct answer: B) s = set()]') 
    
if option != "B) s = set()" and option is not None:
    st.markdown(""" 
             :red[Sorry Wrong answer]\n
             :green[Correct answer is :  s = set()] 
             ## Explanation:
            {} creates an empty dictionary, not a set. 
            set() is the correct way to create an empty set
                     """)
    

