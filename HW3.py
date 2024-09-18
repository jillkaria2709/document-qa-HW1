import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

st.title('My HW3 Question Answering chatbox')

openAImodel = st.sidebar.selectbox("Which model?", ("mini", "regular"))
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
            return soup.get_text()[:500]  # Limiting text to first 500 characters
        except Exception as e:
            return f"Error fetching URL: {e}"

    if url_1:
        st.write("**First URL Content:**")
        st.write(fetch_and_parse(url_1))

    if url_2:
        st.write("**Second URL Content:**")
        st.write(fetch_and_parse(url_2))

if openAImodel == "mini":
    model_to_use = "gpt-4o-mini"
else:
    model_to_use = "gpt-4o"

if 'client' not in st.session_state:
    api_key = st.secrets["openai_key"]
    st.session_state.client = OpenAI(api_key=api_key)

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

    client = st.session_state.client
    stream = client.chat.completions.create(
        model=model_to_use,
        messages=st.session_state.messages,
        stream=True
    )

    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Ensure response is less than 150 words
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
