import os
import openai
from vector_store import create_vector_store, search_similar_chunks
from dotenv import load_dotenv

load_dotenv()  # Load API key dari .env
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGChatbot:
    def __init__(self, documents):
        # Buat vector store dan simpan teksnya
        self.store, self.texts = create_vector_store(documents)

    def ask(self, query):
        # Cari konteks yang relevan di vector store
        context_text = search_similar_chunks(self.store, query)

        # Buat prompt untuk GPT
        prompt = (
            f"Context: {context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Panggil OpenAI ChatCompletion (GPT-3.5 turbo)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )

        # Ambil jawaban dari response
        answer = response.choices[0].message.content.strip()
        return answer
