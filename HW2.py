import streamlit as st
from openai import OpenAI

# Show title and description
st.title("📄 Document Summarizer")
st.write(
    "Upload a document and choose how you want it summarized – GPT will generate a summary! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# OpenAI client
client = OpenAI(api_key=st.secrets["openai_key"])

# File uploader for document
uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

# Sidebar options for summary type
st.sidebar.title("Summary Options")
summary_type = st.sidebar.radio(
    "Select summary type:",
    ("100-word summary", "2 connecting paragraphs", "5 bullet points")
)

use_advanced_model = st.sidebar.checkbox("Use Advanced Model")
model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"

if uploaded_file:
    document = uploaded_file.read().decode()

    if summary_type == "100-word summary":
        instruction = "Summarize the document in 100 words."
    elif summary_type == "2 connecting paragraphs":
        instruction = "Summarize the document in 2 paragraphs."
    else:
        instruction = "Summarize the document in 5 bullet points."

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