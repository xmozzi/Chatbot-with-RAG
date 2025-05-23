import os
import openai
from vector_store import create_vector_store, search_similar_chunks
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGChatbot:
    def __init__(self, documents):
        self.store, self.texts = create_vector_store(documents)

    def ask(self, query):
        context_text = search_similar_chunks(self.store, query)
        prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
        )

        answer = response.choices[0].message.content.strip()
        return answer
