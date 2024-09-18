import streamlit as st
import openai
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import os
from groq import Groq

# Title and description
st.title("📄 My Homework 3 question answering Chatbox")

# Sidebar options
st.sidebar.header("Options")

# Option to input two URLs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")

# Option to pick the LLM vendor (OpenAI, Gemini, Groq)
llm_vendor = st.sidebar.selectbox("Select LLM Vendor", ["OpenAI", "Gemini", "Groq"])

# Option to pick the type of conversation memory
memory_type = st.sidebar.selectbox("Select Conversation Memory Type", ["Buffer of 5 questions", "Conversation Summary", "Buffer of 5,000 tokens"])

# Determine the buffer size based on the selected memory type
if memory_type == "Buffer of 5 questions":
    buffer_size = 5
elif memory_type == "Buffer of 5,000 tokens":
    buffer_size = 5000  
else:
    buffer_size = 10  

# Model selection based on LLM vendor
if llm_vendor == "OpenAI":
    model_to_use = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4", "gpt-3.5-turbo"])
elif llm_vendor == "Gemini":
    model_to_use = st.sidebar.selectbox("Select Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
elif llm_vendor == "Groq":
    model_to_use = st.sidebar.selectbox("Select Groq Model", ["llama3-8b-8192", "other-groq-model"])

# Buffer size slider
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)

# Set up LLM clients based on the vendor
if llm_vendor == "OpenAI":
    openai.api_key = st.secrets["openai_key"]
elif llm_vendor == "Gemini":
    genai.configure(api_key=st.secrets["gemini_api_key"])
elif llm_vendor == "Groq":
    groq_client = Groq(api_key=st.secrets["grok_api_key"])

# A dictionary to store parsed URL content in session state
if "parsed_urls" not in st.session_state:
    st.session_state["parsed_urls"] = {}

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you"}]

# Display chat messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# URL validation function
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Extract text from URL
def extract_text_from_url(url):
    try:
        if url in st.session_state["parsed_urls"]:
            return st.session_state["parsed_urls"][url]  # Return cached content if already parsed
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        st.session_state["parsed_urls"][url] = text  # Cache the parsed content
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL {url}: {e}")
        return ""

# Parse URLs and cache the content in memory
if url1 and is_valid_url(url1):
    extract_text_from_url(url1)
if url2 and is_valid_url(url2):
    extract_text_from_url(url2)

# Function to check if the user's question is related to the URL
def is_question_related_to_url(prompt):
    keywords = ["content", "details", "info from", "link", "URL"]
    return any(keyword in prompt.lower() for keyword in keywords)

# Handling user input
if prompt := st.chat_input("Ask your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Limit the buffer to the specified size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Combine conversation history and URL texts only if the prompt is related to the URL
    if is_question_related_to_url(prompt):
        url_texts = list(st.session_state["parsed_urls"].values())
    else:
        url_texts = []  # No URL content for general questions

    combined_messages = st.session_state.messages + [{"role": "system", "content": "\n".join(url_texts)}]
    
    # OpenAI Response Handling
    if llm_vendor == "OpenAI":
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # Initialize OpenAI client with secret key
        messages = [
            {"role": "system", "content": 'You answer questions about web services.'},
            {"role": "user", "content": prompt}  # Pass the user's prompt as the message content
        ]
        # Call the OpenAI API to get the response
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0  # Adjust temperature if needed
        )
        reply = response.choices[0].message.content  # Get the response content

        with st.chat_message("assistant"):
            st.write(reply)

        # Save the assistant's reply in the session state for conversation history
        st.session_state.messages.append({"role": "assistant", "content": reply})
        
    # Gemini Response Handling
    elif llm_vendor == "Gemini":
        model = genai.GenerativeModel(model_to_use)
        response = model.generate_content("\n".join([msg["content"] for msg in combined_messages]))
        reply = response.text
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    
    # Groq Response Handling
    elif llm_vendor == "Groq":
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model=model_to_use
        )
        reply = chat_completion.choices[0].message.content  # Get the response content

        with st.chat_message("assistant"):
            st.write(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

    # Asking for more information
    more_info_prompt = "Do you want more information?"
    with st.chat_message("assistant"):
        st.write(more_info_prompt)
    st.session_state.messages.append({"role": "assistant", "content": more_info_prompt})

    # Get user input for more information
    if follow_up := st.chat_input("Please type 'yes' or 'no':", key="follow_up"):
        st.session_state.messages.append({"role": "user", "content": follow_up})

        if follow_up.lower() == "yes":
            # If user says 'yes', provide more info
            with st.chat_message("assistant"):
                more_info_response = f"Here's some additional info: {reply}"  
                st.write(more_info_response)
            st.session_state.messages.append({"role": "assistant", "content": more_info_response})
            
            # Ask again after providing more info
            with st.chat_message("assistant"):
                st.write("Do you want help with other questions?")
            st.session_state.messages.append({"role": "assistant", "content": "Do you want help with other questions?"})
        
        elif follow_up.lower() == "no":
            # If user says 'no', ask if they need help with other questions
            next_question_prompt = "Do you want help with other questions?"
            with st.chat_message("assistant"):
                st.write(next_question_prompt)
            st.session_state.messages.append({"role": "assistant", "content": next_question_prompt})

    # Limit messages to buffer size after completing the flow
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size*2:]
