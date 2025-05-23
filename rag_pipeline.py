# rag_pipeline.py
import os
import openai
from vector_store import create_vector_store, search_similar_chunks
from dotenv import load_dotenv

load_dotenv()  # Baca file .env

# Set API key dari env var
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGChatbot:
    def __init__(self, documents):
        # buat vector store dan simpan teksnya
        self.store, self.texts = create_vector_store(documents)

    def ask(self, query):
        # cari konteks yang relevan di vector store
        context_chunks = search_similar_chunks(self.store, query)

        # gabungkan konteks jadi string
        context_text = "\n".join(context_chunks)

        # buat prompt untuk GPT
        prompt = (
            f"Context: {context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # panggil OpenAI ChatCompletion (GPT-3.5 turbo)
       import openai

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.2,
)



        # ambil jawaban dari response
        answer = response.choices[0].message.content.strip()
        return answer
