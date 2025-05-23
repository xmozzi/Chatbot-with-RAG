import streamlit as st
from pdf_loader import load_pdf
from rag_pipeline import RAGChatbot
import os
from dotenv import load_dotenv

load_dotenv()  # load environment variables dari .env

st.set_page_config(page_title="RAG Chatbot with PDF", layout="centered")
st.title("📚 Chat with Your PDF - RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file:
    with st.spinner("Processing document..."):
        documents = load_pdf(uploaded_file)
        st.write(f"Loaded {len(documents)} chunks from PDF.")  # Debug output
        chatbot = RAGChatbot(documents)
    st.success("Document is ready!")

    if query:
        with st.spinner("Generating answer..."):
            try:
                answer = chatbot.ask(query)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error while generating answer: {e}")
