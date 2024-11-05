# database.py
import psycopg2
from config import DBConfig, ModelConfig

class Database:
    @staticmethod
    def setup():
        """Setup database if it doesn't exist."""
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database="postgres"
        )
        conn.autocommit = True
        with conn.cursor() as c:
            c.execute(f"SELECT 1 FROM pg_database WHERE datname='{DBConfig.DATABASE}'")
            if not c.fetchone():
                c.execute(f"CREATE DATABASE {DBConfig.DATABASE}")
        conn.close()

    @staticmethod
    def create_table():
        """Create the vector store table if it doesn't exist."""
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database=DBConfig.DATABASE
        )
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {DBConfig.TABLE_NAME} (
                        id SERIAL PRIMARY KEY,
                        content TEXT,
                        metadata JSONB,
                        embedding VECTOR({ModelConfig.EMBED_DIM}),
                        semantic_labels TEXT[],
                        section_summary TEXT,
                        relevance_scores JSONB,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def initialize_if_empty():
        """Check if the database is empty and needs initialization."""
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database=DBConfig.DATABASE
        )
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {DBConfig.TABLE_NAME}")
                count = cur.fetchone()[0]
                return count == 0
        finally:
            conn.close()

    @staticmethod
    def clear(force=False):
        """Clear the database only if force=True."""
        if not force:
            print("Database clear skipped - use force=True to clear existing data")
            return
            
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database=DBConfig.DATABASE
        )
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {DBConfig.TABLE_NAME}")
            conn.commit()
            print("Database cleared successfully.")
        except Exception as e:
            print(f"Error clearing database: {e}")
        finally:
            conn.close()

    @staticmethod
    def get_document_hash(file_path):
        """Get hash of existing document if it exists."""
        conn = psycopg2.connect(
            host=DBConfig.HOST,
            password=DBConfig.PASSWORD,
            port=DBConfig.PORT,
            user=DBConfig.USER,
            database=DBConfig.DATABASE
        )
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT metadata->>'file_hash' 
                    FROM {DBConfig.TABLE_NAME} 
                    WHERE metadata->>'file_path' = %s 
                    LIMIT 1
                """, (file_path,))
                result = cur.fetchone()
                return result[0] if result else None
        finally:
            conn.close()