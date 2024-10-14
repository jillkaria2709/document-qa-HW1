import streamlit as st

# Set the page title
st.title("Simple Test")

# Simple text output to verify Streamlit is working
st.write("If you see this message, Streamlit is working fine!")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the CSV file
    st.write("CSV file loaded successfully!")
    st.write(df.head())  # Show the first few rows of the data
