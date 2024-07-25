import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader
import google.generativeai as genai
from PyPDF2 import PdfReader
from huggingface_hub import login
import os
import time
from langchain.llms.base import LLM
from typing import Any, List, Optional, Dict
from pydantic import Field
from langchain.schema import Document


class GeminiLLM(LLM):
    model_name: str = Field(..., description="gemini-pro")
    model: Any = Field(None, description="The GenerativeModel instance")
    
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name=model_name)
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
    
load_dotenv(Path(".env"))

def generate_rag_prompt(query, context):
    prompt=("""
    You are Sivi an AI assistant for Apple's new product Apple Vision Pro. Your role is to provide accurate and helpful information about the product.\
    Analyze customer's queries and retrieve relevant information from the Apple Vision Pro's database. Provide concise and informative response to the customer.\
    Use this information to offer insights and statistics when answering questions.\
    When responding to queries, prioritize official Apple vision pro policies and guidelines. If a query falls outside your knowledge base, politely direct the customer to the appropriate representative.\
            QUESTION: '{query}'
            CONTEXT: '{context}'

            Response: 
""").format(query=query,context=context)
    return prompt

def generate_answer(prompt):
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    llm = genai.GenerativeModel(model_name='gemini-pro')
    answer = llm.generate_content(prompt)
    return answer.text

pdfreader = PdfReader("Apple_Vision_Pro_Privacy_Overview.pdf")

def process_additional_data():
    # Extract text from PDF
    pdf_text = ""
    for page in pdfreader.pages:
        pdf_text += page.extract_text()
    
    # Create a Document object from PDF text
    pdf_doc = Document(page_content=pdf_text, metadata={"source": "Apple_Vision_Pro_Privacy_Overview.pdf"})
    
    website_url="https://www.apple.com/apple-vision-pro/"
    web_loader=WebBaseLoader(website_url)
    web_docs = web_loader.load()
    
    youtube_url="https://www.youtube.com/watch?v=TX9qSaGXFyg"
    youtube_loader=YoutubeLoader.from_youtube_url(youtube_url=youtube_url,add_video_info=True)
    youtube_docs = youtube_loader.load()

    all_docs = [pdf_doc] + web_docs + youtube_docs

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)
    texts = [doc.page_content for doc in split_docs]
    embedding_function = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    faiss_vector_store = FAISS.from_texts(texts, embedding_function)

    return faiss_vector_store


st.set_page_config(page_title="Chat with Sivi", layout="wide", )
st.title("Chat with Sivi..!!")

if "data_processed" not in st.session_state:
    st.session_state.data_processed = False

if "faiss_vector_store" not in st.session_state:
    st.session_state.faiss_vector_store = None


with st.spinner("Loading and processing data..."):
        if not st.session_state.data_processed:
            st.session_state.faiss_vector_store = process_additional_data()
            st.session_state.data_processed=True

st.sidebar.markdown("## **Welcome to Apple Vision Pro's Chatbot Sivi**")

def typing_animation(text, speed):
            for char in text:
                yield char
                time.sleep(speed)

if "intro_displayed" not in st.session_state:
    st.session_state.intro_displayed = True
    intro = "Hello, I am Sivi, a Chatbot designed for Apple Vision Pro"
    intro2= "You can chat with Sivi"
    st.write_stream(typing_animation(intro,0.02))
    st.write_stream(typing_animation(intro2,0.02))

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input("Chat with Sivi...")
 
if prompt:
    with st.chat_message('user'):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    query_text = prompt.strip()
    gemini_llm = GeminiLLM(model_name='gemini-pro')
    if st.session_state.faiss_vector_store is not None:
        relevant_docs = st.session_state.faiss_vector_store.similarity_search(query_text, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        prompt = generate_rag_prompt(query=query_text,context=context)
        answer = generate_answer(prompt)
        
        typing_speed = 0.02 
        if "context" or "no" in answer:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer, typing_speed))
        else:
            with st.chat_message("assistant"):
                st.write_stream(typing_animation(answer,typing_speed))
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error("Database not initialized. Kindly reload and upload the PDF first.")