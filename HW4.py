import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

chroma_client = chromadb.PersistentClient()

collection = chroma_client.get_or_create_collection(name="My_Collection")

collection.upsert(
    documents=[
        "This is a document about pineapples",
        "This is a document about oranges"
    ],
    ids=["id1","id2"]
)

results = collection.query(
    query_texts=["This is a query document about florida"],
    n_results= 2
)

print(results)