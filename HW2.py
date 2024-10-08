import os
import streamlit as st
from openai import OpenAI
#import google.generativeai as genai
import requests
from groq import Groq  # Importing Groq API
import time

# Show title and description
st.title("📄 Document Summarizer")
st.write(
    "Upload a document or provide a URL and choose how you want it summarized – GPT, Gemini, Groq, or OpenRouter will generate a summary! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# OpenAI client configuration
openai_client = OpenAI(api_key=st.secrets["openai_key"])

# Gemini API configuration
#genai.configure(api_key=st.secrets["gemini_api_key"])

# Groq API configuration
grok_client = Groq(api_key=st.secrets["grok_api_key"])

# Sidebar options for input method
input_method = st.sidebar.radio("Select input method:", ("Upload a document", "Enter a URL"))

# Sidebar options for summary type
st.sidebar.title("Summary Options")
summary_type = st.sidebar.radio(
    "Select summary type:",
    ("100-word summary", "2 paragraphs", "5 bullet points")
)

# Language selection dropdown
language = st.sidebar.selectbox(
    "Select output language:",
    ("English", "French", "Spanish")
)

# Checkbox for advanced model usage
use_advanced_model = st.sidebar.checkbox("Use Advanced OpenAI Model")

# Model selection between OpenAI, Gemini, Groq, and OpenRouter
model_provider = st.sidebar.radio("Select AI Model Provider:", ("OpenAI", "Gemini", "Groq", "OpenRouter"))

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

# Function to call OpenRouter API
def call_openrouter_api(api_key, document, instruction):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "mattshumer/reflection-70b:free",  # Example model
            "messages": [
                {"role": "user", "content": f"Here's a document: {document} \n\n---\n\n {instruction}"},
            ],
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No completion found.")
        else:
            return f"OpenRouter API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling OpenRouter API: {e}"

# Variable to hold the document content
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
    elif summary_type == "2 paragraphs":
        instruction = f"Summarize the document in 2 paragraphs in {language}."
    else:
        instruction = f"Summarize the document in 5 bullet points in {language}."

    # Using OpenAI, Gemini, Groq, or OpenRouter based on user choice
    if model_provider == "OpenAI":
        model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {instruction}",
            }
        ]

        # Stream OpenAI response
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)

    #elif model_provider == "Gemini":
        #model = genai.GenerativeModel('gemini-1.5-flash-latest')
        #gemini_response = model.generate_content(f"Here's a document: {document} \n\n---\n\n {instruction}")
        #st.write(f"**Response from Gemini:**\n{gemini_response.text}")

    elif model_provider == "Groq":
        success = False
        while not success:
            try:
                chat_completion = grok_client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Here's a document: {document} \n\n---\n\n {instruction}"
                        }
                    ],
                    model="llama3-8b-8192"
                )
                st.write(f"**Response from Groq:**\n{chat_completion.choices[0].message.content}")
                success = True  # Exit loop when successful
            except Exception as e:  # Catching general exception
                st.error(f"An error occurred: {e}")
                if "rate_limit" in str(e).lower():
                    retry_after_seconds = 60  # Default retry time
                    st.warning(f"Rate limit exceeded. Retrying in {retry_after_seconds} seconds.")
                    time.sleep(retry_after_seconds)
                else:
                    break  # Exit if other errors occur

    elif model_provider == "OpenRouter":
        api_key = st.secrets["openrouter_api_key"]
        openrouter_response = call_openrouter_api(api_key, document, instruction)
        st.write(f"**Response from OpenRouter:**\n{openrouter_response}")

else:
    st.write("Please upload a document or enter a URL to summarize.")
