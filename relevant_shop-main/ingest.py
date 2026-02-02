"""
Скрипт для парсинга базы товаров и сохранения в векторное хранилище ChromaDB
"""
import re
import os
import shutil
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import OpenRouterEmbeddings

# Загружаем переменные окружения из .env
load_dotenv()


def parse_products_from_markdown(file_path: str) -> List[Dict[str, str]]:
    """
    Парсит файл base.md и извлекает информацию о товарах

    Args:
        file_path: путь к файлу с товарами

    Returns:
        Список словарей с информацией о товарах
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разделяем по товарам (они разделены ---)
    products_raw = content.split('---')
    products = []

    for product_text in products_raw:
        # Пропускаем заголовок и пустые блоки
        if not product_text.strip() or 'База товаров' in product_text:
            continue

        # Извлекаем поля с помощью регулярных выражений
        product = {}

        # ID
        id_match = re.search(r'\*\*ID:\*\*\s*(.+)', product_text)
        if id_match:
            product['id'] = id_match.group(1).strip()

        # Название
        name_match = re.search(r'\*\*Название:\*\*\s*(.+)', product_text)
        if name_match:
            product['name'] = name_match.group(1).strip()

        # Категория
        category_match = re.search(r'\*\*Категория:\*\*\s*(.+)', product_text)
        if category_match:
            product['category'] = category_match.group(1).strip()

        # Цена
        price_match = re.search(r'\*\*Цена:\*\*\s*(.+)', product_text)
        if price_match:
            product['price'] = price_match.group(1).strip()

        # Описание
        description_match = re.search(r'\*\*Описание:\*\*\s*(.+?)(?=\*\*Характеристики:\*\*)', product_text, re.DOTALL)
        if description_match:
            product['description'] = description_match.group(1).strip()

        # Характеристики
        features_match = re.search(r'\*\*Характеристики:\*\*\s*(.+)', product_text, re.DOTALL)
        if features_match:
            product['features'] = features_match.group(1).strip()

        # Добавляем товар, если у него есть ID и название
        if 'id' in product and 'name' in product:
            products.append(product)

    return products


def create_documents_from_products(products: List[Dict[str, str]]) -> List[Document]:
    """
    Создает Document объекты из списка товаров для ChromaDB

    Args:
        products: список товаров

    Returns:
        Список Document объектов
    """
    documents = []

    for product in products:
        # Создаем текст для векторизации (описание + характеристики)
        page_content = f"{product.get('name', '')}\n\n"
        page_content += f"{product.get('description', '')}\n\n"
        page_content += f"Характеристики: {product.get('features', '')}"

        # Метаданные для фильтрации
        metadata = {
            'id': product.get('id', ''),
            'name': product.get('name', ''),
            'category': product.get('category', ''),
            'price': product.get('price', ''),
        }

        documents.append(Document(
            page_content=page_content,
            metadata=metadata
        ))

    return documents


def vectorize_and_save_to_chroma(
    documents: List[Document],
    collection_name: str = "products",
    persist_directory: str = "./chroma_db"
):
    """
    Создает векторные представления товаров и сохраняет в ChromaDB

    Args:
        documents: список Document объектов
        collection_name: название коллекции в ChromaDB
        persist_directory: директория для сохранения базы данных
    """
    # Удаляем старую базу данных, если она существует
    if os.path.exists(persist_directory):
        print(f"Удаление старой базы данных из {persist_directory}...")
        shutil.rmtree(persist_directory)
        print(f"✓ Старая база данных удалена")

    print(f"Загрузка модели эмбеддингов Google через OpenRouter...")

    # Используем кастомный класс для Google gemini-embedding-001 через OpenRouter
    embeddings = OpenRouterEmbeddings(
        model="google/gemini-embedding-001",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        site_url="http://localhost",
        site_name="Product Search"
    )

    print(f"Создание векторного хранилища ChromaDB...")

    # Создаем векторное хранилище (старая база уже удалена)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    print(f"✓ Векторизация завершена!")
    print(f"✓ Сохранено {len(documents)} товаров в коллекцию '{collection_name}'")
    print(f"✓ База данных сохранена в: {persist_directory}")

    return vectorstore


def main():
    """Основная функция"""
    # Путь к файлу с товарами
    products_file = Path(__file__).parent / "products" / "base.md"

    print("=" * 60)
    print("Векторизация базы товаров для системы рекомендаций")
    print("=" * 60)

    # 1. Парсим товары из markdown файла
    print(f"\n1. Парсинг товаров из {products_file}...")
    products = parse_products_from_markdown(str(products_file))
    print(f"✓ Найдено {len(products)} товаров")

    # Выводим первый товар для проверки
    if products:
        print(f"\nПример первого товара:")
        print(f"  ID: {products[0].get('id')}")
        print(f"  Название: {products[0].get('name')}")
        print(f"  Категория: {products[0].get('category')}")
        print(f"  Цена: {products[0].get('price')}")

    # 2. Создаем документы для ChromaDB
    print(f"\n2. Создание документов для векторного хранилища...")
    documents = create_documents_from_products(products)
    print(f"✓ Создано {len(documents)} документов")

    # 3. Векторизация и сохранение в ChromaDB
    print(f"\n3. Векторизация и сохранение в ChromaDB...")
    vectorstore = vectorize_and_save_to_chroma(
        documents=documents,
        collection_name="products",
        persist_directory=str(Path(__file__).parent / "chroma_db")
    )

    print("\n" + "=" * 60)
    print("Готово! Векторная база данных создана.")
    print("=" * 60)


if __name__ == "__main__":
    main()