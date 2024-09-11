import streamlit as st
from openai import OpenAI
import requests

# Show title and description
st.title("📄 Document Summarizer")
st.write(
    "Upload a document or provide a URL and choose how you want it summarized – GPT will generate a summary! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])

# Sidebar options for input method
input_method = st.sidebar.radio("Select input method:", ("Upload a document", "Enter a URL"))

# Sidebar options for summary type
st.sidebar.title("Summary Options")
summary_type = st.sidebar.radio(
    "Select summary type:",
    ("100-word summary", "2 connecting paragraphs", "5 bullet points")
)

# Language selection dropdown
language = st.sidebar.selectbox(
    "Select output language:",
    ("English", "French", "Spanish")
)

# Checkbox for advanced model usage
use_advanced_model = st.sidebar.checkbox("Use Advanced Model")
model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

# Function to fetch content from a URL
def fetch_content_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            st.error("Failed to fetch content from the URL. Please check the URL and try again.")
            return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

document = None

if input_method == "Upload a document":
    # File uploader for document
    uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))
    if uploaded_file:
        document = uploaded_file.read().decode()

elif input_method == "Enter a URL":
    # URL input field
    url = st.text_input("Enter a URL to fetch content:")
    if url:
        document = fetch_content_from_url(url)

# Proceed with summary generation if there's a document
if document:
    if summary_type == "100-word summary":
        instruction = f"Summarize the document in 100 words in {language}."
    elif summary_type == "2 connecting paragraphs":
        instruction = f"Summarize the document in 2 paragraphs in {language}."
    else:
        instruction = f"Summarize the document in 5 bullet points in {language}."

    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )

    st.write_stream(stream)
else:
    st.write("Please upload a document or enter a URL to summarize.")
