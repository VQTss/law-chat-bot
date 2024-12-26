import sqlite3
import pickle

# Connect to SQLite
def connect_db(db_path="embeddings.db"):
    conn = sqlite3.connect(db_path)
    return conn

# Initialize table
def init_db(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT,
        content TEXT,
        embedding BLOB
    )
    """)
    conn.commit()

# Insert data
def insert_embedding(conn, document_id, content, embedding):
    cursor = conn.cursor()
    embedding_blob = pickle.dumps(embedding)
    cursor.execute("""
    INSERT INTO embeddings (document_id, content, embedding)
    VALUES (?, ?, ?)
    """, (document_id, content, embedding_blob))
    conn.commit()

# Retrieve data
def fetch_all_embeddings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT document_id, content, embedding FROM embeddings")
    return cursor.fetchall()
