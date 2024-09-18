import streamlit as st
from openai import OpenAI
import requests

# Initialize OpenAI client based on selected model
def initialize_openai_client(api_key, model_to_use):
    return OpenAI(api_key=api_key, model=model_to_use)

# Initialize Gemini client (pseudo-code, replace with actual Gemini initialization)
def initialize_gemini_client(api_key):
    return OpenAI(api_key=api_key, model="gemini")

# Function to fetch data from URL
def fetch_url_content(url):
    try:
        response = requests.get(url)
        return response.text[:500]  # Limit the content fetched from URL to avoid overflow
    except Exception as e:
        return f"Failed to fetch content from {url}: {e}"

st.title('My LAB3 Question Answering Chatbox')

# Sidebar: Model selection
llm_vendor = st.sidebar.radio("Select LLM Vendor", ("OpenAI", "Gemini", "Hermes"))
openAImodel = st.sidebar.radio("OpenAI Models", ("gpt-3.5", "gpt-4o"), disabled=llm_vendor != "OpenAI")

# Sidebar: URLs
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")

# Sidebar: Buffer size and conversation memory type
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)
memory_type = st.sidebar.radio("Conversation Memory", ["Short-term", "Long-term"])

# Initialize the correct LLM client based on user selection
if 'client' not in st.session_state:
    api_key = st.secrets["openai_key"] if llm_vendor == "OpenAI" else st.secrets["openrouter_key"]
    
    if llm_vendor == "OpenAI":
        model_to_use = openAImodel
        st.session_state.client = initialize_openai_client(api_key, model_to_use)
    elif llm_vendor == "Gemini":
        st.session_state.client = initialize_gemini_client(api_key)

# Manage conversation history
if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help?"}]

# Display all previous messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Input prompt
if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Maintain the buffer size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    # Fetch and combine content from URLs
    url1_content = fetch_url_content(url1) if url1 else ""
    url2_content = fetch_url_content(url2) if url2 else ""

    # Combine prompt with URL contents
    prompt_with_urls = prompt + "\n\nURL1 Content:\n" + url1_content + "\n\nURL2 Content:\n" + url2_content

    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream the LLM response
    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use if llm_vendor == "OpenAI" else "other_model",
        messages=st.session_state.messages,
        stream=True
    )

    # Display the streamed response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Limit the response to 150 words
    response = response[:150]
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Automatically ask for more information
    more_info_question = "Want more info? (Yes/No)"
    st.session_state.messages.append({"role": "assistant", "content": more_info_question})

# Handle the user's response for more information
if prompt and prompt.lower() in ["yes", "no"]:
    if prompt.lower() == "yes":
        st.session_state.messages.append({"role": "assistant", "content": "Continuing..."})
    elif prompt.lower() == "no":
        st.session_state.messages.append({"role": "assistant", "content": "What else?"})