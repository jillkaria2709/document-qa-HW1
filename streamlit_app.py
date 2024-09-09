import streamlit as st
import os

# Sidebar with buttons for navigation
st.sidebar.title("Navigation")
home_button = st.sidebar.button("Home")
lab1_button = st.sidebar.button("HW1")
lab2_button = st.sidebar.button("H12")

# Initialize a default page (Home) to be shown if no button is clicked yet
if 'page' not in st.session_state:
    st.session_state.page = 'HW2'

# Update page state based on button clicks
if home_button:
    st.session_state.page = 'Home'
elif lab1_button:
    st.session_state.page = 'HW1'
elif lab2_button:
    st.session_state.page = 'HW2'

# Display the appropriate content based on the current page
if st.session_state.page == 'Home':
    st.title("Welcome to my homeworks")
    st.write("You can see my homeworks here.")
elif st.session_state.page == 'HW1':
    st.title("HW1")
    if os.path.exists('HW1.py'):
        with open('HW1.py') as f:
            code = f.read()
            exec(code)  # Executes the LAB1.py code
    else:
        st.error("HW1.py file not found.")
elif st.session_state.page == 'HW2':
    st.title("HW2")
    if os.path.exists('HW2.py'):
        with open('HW2.py') as f:
            code = f.read()
            exec(code)  # Executes the LAB2.py code
    else:
        st.error("HW2.py file not found.")
