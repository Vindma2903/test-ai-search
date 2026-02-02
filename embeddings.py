"""
Модуль для работы с эмбеддингами через OpenRouter API
"""
import os
from typing import List
from openai import OpenAI
from langchain_core.embeddings import Embeddings


class OpenRouterEmbeddings(Embeddings):
    """Кастомный класс для работы с OpenRouter Embeddings через OpenAI SDK"""

    def __init__(
        self,
        model: str = "google/gemini-embedding-001",
        api_key: str = None,
        site_url: str = "http://localhost",
        site_name: str = "Product Search"
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.site_url = site_url
        self.site_name = site_name

        # Проверяем наличие API ключа
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY не найден в переменных окружения")

        # Создаем OpenAI клиент для OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Векторизует список документов"""
        embedding = self.client.embeddings.create(
            extra_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return [item.embedding for item in embedding.data]

    def embed_query(self, text: str) -> List[float]:
        """Векторизует один запрос"""
        embedding = self.client.embeddings.create(
            extra_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return embedding.data[0].embedding