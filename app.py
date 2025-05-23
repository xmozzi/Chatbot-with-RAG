import streamlit as st
from pdf_loader import load_pdf
from rag_pipeline import RAGChatbot

st.set_page_config(page_title="RAG Chatbot with PDF", layout="centered")
st.title("📚 Chat with Your PDF - RAG Chatbot")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
query = st.text_input("Ask a question about the document:")

if uploaded_file:
    with st.spinner("Processing document..."):
        documents = load_pdf(uploaded_file)
        chatbot = RAGChatbot(documents)
    st.success("Document is ready!")

    if query:
        with st.spinner("Generating answer..."):
            answer = chatbot.ask(query)
        st.markdown(f"**Answer:** {answer}")