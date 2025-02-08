import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def check_table_exists(cur, table_name):
    """Check if a table exists in the database."""
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = %s
        );
    """, (table_name,))
    return cur.fetchone()[0]

def create_tables():
    # Database connection parameters
    db_params = {
        'dbname': 'sorcerer',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }

    # Connect to the database
    conn = psycopg2.connect(**db_params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    try:
        # Create pgvector extension if it doesn't exist
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Drop existing table if it exists
        if check_table_exists(cur, 'book_embeddings'):
            print("Dropping existing book_embeddings table...")
            cur.execute("DROP TABLE book_embeddings;")
            print("Existing table dropped.")

        # Create table for book embeddings
        print("Creating new book_embeddings table...")
        cur.execute("""
            CREATE TABLE book_embeddings (
                id SERIAL PRIMARY KEY,
                book_name VARCHAR(255) NOT NULL,
                embedding vector(1536),
                tokens TEXT[],
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        print("Tables created successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_tables() 