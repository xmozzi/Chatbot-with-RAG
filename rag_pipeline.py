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
        # cari konteks yang relevan di vector store (hasilnya string, sesuai kode vector_store.py)
        context_text = search_similar_chunks(self.store, query)

        # buat pesan untuk OpenAI ChatCompletion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"}
        ]

        # panggil OpenAI ChatCompletion (GPT-3.5 turbo)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )

        # ambil jawaban dari response
        answer = response.choices[0].message.content.strip()
        return answer
