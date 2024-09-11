import streamlit as st

# Define individual pages for homework 1 and homework 2
hw1_page = st.Page("HW1.py", title="HW1")
hw2_page = st.Page("HW2.py", title="HW2")

# Initialize navigation with the pages
pg = st.navigation([hw1_page, hw2_page])

# Set page configuration (optional but helps with page title and icon)
st.set_page_config(page_title="Homework Manager", page_icon=":memo:")

# Run the navigation system
pg.run()
