import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

openai_client = chromadb.PersistentClient()

if 'openai_client' not in st.session_state:
    api_key = st.secrets("openai_key")
    st.session_state.openai_client = OpenAI(api_key=api_key)

def add_to_collection(collection,text,filename):
    