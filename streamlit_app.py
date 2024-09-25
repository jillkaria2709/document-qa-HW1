import streamlit as st
from streamlit import session_state as state

# Define individual pages for homework 1 and homework 2
hw42_page = st.Page("HW4-2.py", title="HW4")
hw4_page = st.Page("HW4.py", title="HW4")
hw3_page = st.Page("HW3.py", title="HW3")
hw2_page = st.Page("HW2.py", title="HW2")
hw1_page = st.Page("HW1.py", title="HW1")

# Initialize navigation with the pages
pg = st.navigation([hw42_page,hw4_page,hw3_page,hw2_page,hw1_page])

# Set page configuration (optional but helps with page title and icon)
st.set_page_config(page_title="Homework Manager", page_icon=":memo:")

# Run the navigation system
pg.run()
