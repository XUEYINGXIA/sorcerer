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

def create_vector_table():
    # Database connection parameters
    db_params = {
        "dbname": "sorcerer",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432"
    }
    
    # Connect to PostgreSQL
    conn = psycopg2.connect(**db_params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    try:
        # Enable the vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("Vector extension enabled.")
        
        # Check if table exists
        if check_table_exists(cur, 'text_embeddings'):
            print("Table 'text_embeddings' already exists.")
            return
        
        # Create the table for storing text embeddings
        print("Creating 'text_embeddings' table...")
        cur.execute("""
            CREATE TABLE text_embeddings (
                id SERIAL PRIMARY KEY,
                book_name VARCHAR(255),
                chunk_index INTEGER,
                text_chunk TEXT,
                embedding vector(1536)
            );
        """)
        
        # Create an index for faster similarity searches
        print("Creating vector similarity index...")
        cur.execute("""
            CREATE INDEX embedding_idx 
            ON text_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        print("Successfully created table and index!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    create_vector_table() 