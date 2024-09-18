import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai  # For Gemini

st.title('My HW3 Question Answering chatbox')

# Sidebar to select the LLM vendor (Gemini, OpenRouter, OpenAI)
model_provider = st.sidebar.selectbox("Select LLM Vendor", ("OpenAI", "Gemini", "OpenRouter"))
openAImodel = st.sidebar.selectbox("Which OpenAI model?", ("mini", "regular"))
buffer_size = st.sidebar.slider("Buffer Size", min_value=1, max_value=10, value=2, step=1)

# Input fields for two URLs
url_1 = st.sidebar.text_input("Enter the first URL:")
url_2 = st.sidebar.text_input("Enter the second URL:")

# Button to fetch and parse the URLs
if st.sidebar.button("Fetch URLs"):
    def fetch_and_parse(url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            return soup.get_text()  # Store full text
        except Exception as e:
            return f"Error fetching URL: {e}"

    # Store parsed content in session_state
    if url_1:
        st.session_state['url_1_content'] = fetch_and_parse(url_1)

    if url_2:
        st.session_state['url_2_content'] = fetch_and_parse(url_2)

# Button to display the content of the first URL
if 'url_1_content' in st.session_state:
    if st.sidebar.button("Print First URL Studied"):
        st.write("**First URL Content:**")
        st.write(st.session_state['url_1_content'][:500])  # Limiting output to 500 characters

# API Configuration based on model selection
if model_provider == "OpenAI":
    if openAImodel == "mini":
        model_to_use = "gpt-4o-mini"
    else:
        model_to_use = "gpt-4o"

    if 'client' not in st.session_state:
        api_key = st.secrets["openai_key"]
        st.session_state.client = OpenAI(api_key=api_key)

    client = st.session_state.client

elif model_provider == "Gemini":
    genai.configure(api_key=st.secrets["gemini_api_key"])
    model_to_use = "gemini-1.5-flash-latest"

elif model_provider == "OpenRouter":
    model_to_use = "mattshumer/reflection-70b:free"  # Example model
    api_key = st.secrets["openrouter_api_key"]

# Existing chatbox logic
if 'messages' not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help?"}]

# Display all messages
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Input prompt
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Maintain the buffer size
    if len(st.session_state.messages) > buffer_size * 2:
        st.session_state.messages = st.session_state.messages[-buffer_size * 2:]

    with st.chat_message("user"):
        st.markdown(prompt)

    if model_provider == "OpenAI":
        stream = client.chat.completions.create(
            model=model_to_use,
            messages=st.session_state.messages,
            stream=True
        )

    elif model_provider == "Gemini":
        gemini_response = genai.generate_content(f"Here's a document: {st.session_state['url_1_content'][:1000]} \n\n {prompt}")
        st.session_state.messages.append({"role": "assistant", "content": gemini_response.result})
        st.write(gemini_response.result)

    elif model_provider == "OpenRouter":
        def call_openrouter_api(api_key, document, instruction):
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": model_to_use,
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

        openrouter_response = call_openrouter_api(api_key, st.session_state['url_1_content'][:1000], prompt)
        st.session_state.messages.append({"role": "assistant", "content": openrouter_response})
        st.write(openrouter_response)

    # Automatically ask for more information
    more_info_question = "Want more info? (Yes/No)"
    st.session_state.messages.append({"role": "assistant", "content": more_info_question})

# Handle the user's response for more information
if prompt and prompt.lower() in ["yes", "no"]:
    if prompt.lower() == "yes":
        st.session_state.messages.append({"role": "assistant", "content": "Continuing..."})
    elif prompt.lower() == "no":
        st.session_state.messages.append({"role": "assistant", "content": "What else?"})
