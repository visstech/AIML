

import streamlit as st, datetime
# import os ; os.remove('chat.txt') 
# Uncomment the line above to delete the file containing
# all messages, and start over with a fresh chat room.

col1, col2=st.columns([2,3])
with col2:
  with open('chat.txt', 'a+') as file: pass
  with open('chat.txt', 'r+') as file:
    msg=file.read()
    st.text_area('msg', msg, height=350, label_visibility='collapsed')

with col1:
  with st.form('New Message', clear_on_submit=True):
    name=st.text_input('Name')
    message=st.text_area('Message') 
    timestamp=datetime.datetime.now()
    if st.form_submit_button('Add Message'):
      newmsg=f'---  {name}   {timestamp}\n\n{message}\n\n{msg}'
      with open('chat.txt', 'w') as file:
        file.write(newmsg)
      #st.experimental_rerun()
      st.balloons()