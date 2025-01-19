import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import chromadb.api
import sqlite3
chromadb.api.client.SharedSystemClient.clear_system_cache()
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=OllamaEmbeddings(model="all-MiniLM-L6-v2")
st.title("Conversational  RAG with pdf uploads and chat history")
st.write("Upload pdf's and chat write their content")
api_key=st.text_input("enter your groq api key:",type="password")
if api_key:
    llm=ChatGroq(api_key=api_key,model="Gemma2-9b-It")

    session_id=st.text_input("Session ID",value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose A PDF file",type="pdf",accept_multiple_files=False)
    if uploaded_files:
        document=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                loader=PyPDFLoader(temppdf)
                doc=loader.load()
                document.extend(doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
        split = text_splitter.split_documents(document)
        vectorstore=Chroma.from_documents(documents=split,embedding=embeddings)
        retriver = vectorstore.as_retriever()