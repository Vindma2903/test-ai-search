"""
Скрипт для поиска похожих товаров по запросу с Gradio интерфейсом
"""
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embeddings import OpenRouterEmbeddings
import gradio as gr

# Загружаем переменные окружения из .env
load_dotenv()


def load_vectorstore(
    collection_name: str = "products",
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """
    Загружает существующее векторное хранилище ChromaDB

    Args:
        collection_name: название коллекции в ChromaDB
        persist_directory: директория с базой данных

    Returns:
        Объект Chroma векторного хранилища
    """
    print(f"Загрузка модели эмбеддингов Google через OpenRouter...")

    # Используем кастомный класс для Google gemini-embedding-001 через OpenRouter
    embeddings = OpenRouterEmbeddings(
        model="google/gemini-embedding-001",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        site_url="http://localhost",
        site_name="Product Search"
    )

    print(f"Загрузка векторного хранилища из {persist_directory}...")

    # Загружаем существующее хранилище
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    print(f"✓ Векторное хранилище загружено")

    return vectorstore


# Глобальная переменная для хранения vectorstore
print("=" * 60)
print("Инициализация системы поиска похожих товаров")
print("=" * 60)
persist_directory = str(Path(__file__).parent / "chroma_db")
VECTORSTORE = load_vectorstore(
    collection_name="products",
    persist_directory=persist_directory
)
print("✓ Система готова к работе\n")


def search_products(query: str, num_results: int = 5) -> str:
    """
    Функция для Gradio: ищет похожие товары по запросу

    Args:
        query: текст запроса пользователя
        num_results: количество результатов

    Returns:
        HTML с результатами поиска
    """
    if not query or not query.strip():
        return "<p style='color: red;'>Пожалуйста, введите запрос для поиска товаров.</p>"

    try:
        # Выполняем поиск с запасом +1, на случай если запрос совпадает с названием товара
        results = VECTORSTORE.similarity_search(query, k=num_results + 1)

        if not results:
            return "<p style='color: orange;'>Товары не найдены. Попробуйте изменить запрос.</p>"

        # Фильтруем результаты: исключаем товары, название которых совпадает с запросом
        query_lower = query.lower().strip()
        filtered_results = []

        for doc in results:
            name = doc.metadata.get('name', '')
            # Исключаем товар, если его название точно совпадает с запросом
            if name.lower().strip() != query_lower:
                filtered_results.append(doc)
                # Ограничиваем количество результатов
                if len(filtered_results) >= num_results:
                    break

        if not filtered_results:
            return "<p style='color: orange;'>Похожие товары не найдены. Попробуйте изменить запрос.</p>"

        # Формируем HTML с результатами
        html = f"<h3>Найдено {len(filtered_results)} похожих товаров:</h3>"

        for i, doc in enumerate(filtered_results, 1):
            name = doc.metadata.get('name', 'Без названия')
            product_id = doc.metadata.get('id', 'N/A')
            category = doc.metadata.get('category', 'N/A')
            price = doc.metadata.get('price', 'N/A')

            html += f"""
            <div style='border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; background: #f9f9f9;'>
                <h4 style='margin: 0 0 10px 0; color: #333;'>{i}. {name}</h4>
                <p style='margin: 5px 0;'><strong>ID:</strong> {product_id}</p>
                <p style='margin: 5px 0;'><strong>Категория:</strong> {category}</p>
                <p style='margin: 5px 0;'><strong>Цена:</strong> {price}</p>
            </div>
            """

        return html

    except Exception as e:
        return f"<p style='color: red;'>Ошибка при поиске: {str(e)}</p>"


# Создаем Gradio интерфейс
def create_interface():
    """Создает и возвращает Gradio интерфейс"""

    with gr.Blocks(title="Поиск похожих товаров", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Поиск похожих товаров")
        gr.Markdown("Введите описание товара, который вы ищете, и система найдет наиболее подходящие варианты из базы данных.")

        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Запрос",
                    placeholder="Например: смартфон с хорошей камерой и долгой батареей",
                    lines=3
                )

            with gr.Column(scale=1):
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Количество результатов"
                )

        search_btn = gr.Button("Найти товары", variant="primary", size="lg")

        results_output = gr.HTML(label="Результаты")

        # Примеры запросов
        gr.Examples(
            examples=[
                ["Смартфон с хорошей камерой", 5],
                ["Ноутбук для работы и игр", 5],
                ["Беспроводные наушники", 3],
                ["Кухонная техника", 5],
            ],
            inputs=[query_input, num_results],
        )

        # Обработчик кнопки
        search_btn.click(
            fn=search_products,
            inputs=[query_input, num_results],
            outputs=results_output
        )

        # Также поиск при нажатии Enter
        query_input.submit(
            fn=search_products,
            inputs=[query_input, num_results],
            outputs=results_output
        )

    return demo


if __name__ == "__main__":
    # Запускаем Gradio приложение
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )