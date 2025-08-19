import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI



OPENAI_API_KEY = "sk-proj-hfjbhalihna456sSTHsth8gdfdffddghLHFI-PrSI9SEa8_Txw0U9WtejgUFZrbeDFK56HOOeuw2xghoTxH7h7JNahR2R3nGnZaQPhEA"

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

    # embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Use a free local model instead of OpenAI
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Creating Vector Store

    # vector_store = FAISS.from_texts(chunks, embeddings)

    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user Question

    user_question = st.text_input('Enter your Question')

    # Do similarity Search

    if user_question:
        match = vector_store.similarity_search(user_question)
        st.write(match)

        # Define the llm
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = 'gpt-4.5-turbo'
        )

        # Output Result

        chain = load_qa_chain(llm, chain_type='stuff')
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)