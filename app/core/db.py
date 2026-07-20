import sqlite3
import os

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "story_forge.db")

def get_db_connection():
    """Get a SQLite database connection with row factory and performance pragmas configured."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Enable WAL mode for better concurrency and write performance
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.Error:
        pass
        
    return conn

def init_db():
    """Initialize the database directory, schemas, and performance indexes."""
    # Ensure data directory exists only at initialization time
    os.makedirs(DB_DIR, exist_ok=True)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Create indexes on username and email for O(1) auth lookup speeds
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);")
    
    conn.commit()
    conn.close()
