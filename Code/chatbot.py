import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS



OPENAI_API_KEY = "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"

# Upload PDF File

st.header('Chat Bot ChatGPT')

with st.sidebar:
    st.title('Upload PDF')
    file = st.file_uploader('Upload PDF', type='pdf')

# Extracting  the text

if file is not None:
    pdf_Reader = PdfReader(file)
    text = ''
    for page in pdf_Reader.pages:
        text += page.extract_text()
        # st.write(text)

# Breaking into Chunks

    text_splitter = RecursiveCharacterTextSplitter(
        separators =['\n'],
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

# Generating Embeddings

embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

# Creating Vector Store

vector_store = FAISS.from_texts(chunks, embeddings)