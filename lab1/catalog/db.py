"""Локальный каталог товаров в SQLite."""

import sqlite3
from pathlib import Path

DEFAULT_DATABASE = Path(__file__).resolve().parents[1] / "data" / "products.sqlite"

# name, brand, category, price, stock.
# Цены подобраны так, чтобы одни запросы возвращали несколько строк, а другие — ни одной.
SEED: tuple[tuple[str, str, str, int, int], ...] = (
    ("Acer Aspire 3", "Acer", "laptop", 45000, 6),
    ("MSI Modern 15", "MSI", "laptop", 48000, 4),
    ("Huawei MateBook D16", "Huawei", "laptop", 54000, 7),
    ("Honor MagicBook 14", "Honor", "laptop", 62000, 4),
    ("Lenovo IdeaPad 5", "Lenovo", "laptop", 69000, 3),
    ("ASUS Vivobook 16", "ASUS", "laptop", 89000, 5),
    ("Apple MacBook Air", "Apple", "laptop", 120000, 2),
    ("Dell Inspiron 15", "Dell", "laptop", 150000, 1),
    ("Tecno Spark 20", "Tecno", "smartphone", 12000, 15),
    ("Realme Note", "Realme", "smartphone", 15000, 12),
    ("Honor X9b", "Honor", "smartphone", 25000, 9),
    ("Samsung Galaxy A35", "Samsung", "smartphone", 28000, 8),
    ("Google Pixel 8a", "Google", "smartphone", 45000, 5),
    ("Xiaomi 14", "Xiaomi", "smartphone", 72000, 4),
    ("Samsung Galaxy S24", "Samsung", "smartphone", 95000, 2),
    ("iPhone 15", "Apple", "smartphone", 110000, 3),
    ("Lenovo Tab M10", "Lenovo", "tablet", 16000, 11),
    ("Honor Pad X8", "Honor", "tablet", 18000, 10),
    ("Huawei MatePad SE", "Huawei", "tablet", 22000, 8),
    ("Samsung Galaxy Tab A9", "Samsung", "tablet", 27000, 6),
    ("Xiaomi Pad 6", "Xiaomi", "tablet", 30000, 5),
    ("Huawei MatePad 11", "Huawei", "tablet", 54000, 4),
    ("iPad Air", "Apple", "tablet", 79000, 3),
    ("Samsung Galaxy Tab S9", "Samsung", "tablet", 98000, 2),
)

COLUMNS = ("id", "name", "brand", "category", "price", "stock")


def connect(database: Path | None = None) -> sqlite3.Connection:
    path = database or DEFAULT_DATABASE
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    init_database(connection)
    return connection


def init_database(connection: sqlite3.Connection) -> None:
    existing = [row[1] for row in connection.execute("PRAGMA table_info(products)")]
    if existing and tuple(existing) != COLUMNS:
        connection.execute("DROP TABLE products")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            brand TEXT NOT NULL,
            category TEXT NOT NULL CHECK (category IN ('laptop', 'smartphone', 'tablet')),
            price INTEGER NOT NULL CHECK (price >= 0),
            stock INTEGER NOT NULL CHECK (stock >= 0)
        )
        """
    )
    count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if count != len(SEED):
        connection.execute("DELETE FROM products")
        connection.executemany(
            "INSERT INTO products (name, brand, category, price, stock) VALUES (?, ?, ?, ?, ?)",
            SEED,
        )
    connection.commit()
