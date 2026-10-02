import os
import asyncpg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")

class DatabaseManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        """Initializes the async connection pool to Supabase."""
        if not DATABASE_URL:
            raise ValueError("FATAL: SUPABASE_DATABASE_URL is missing from environment variables.")
        
        # Create a connection pool (min 5, max 20 connections)
        self.pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        print("✅ Database connection pool established.")

    async def disconnect(self):
        """Closes the connection pool securely."""
        if self.pool:
            await self.pool.close()
            print("🛑 Database connection pool closed.")

# Singleton instance to be used across the app
db = DatabaseManager()

# FastAPI Dependency Injection
async def get_db_connection():
    """Yields a database connection for individual route execution."""
    if not db.pool:
        raise Exception("Database pool is not initialized.")
    async with db.pool.acquire() as connection:
        yield connection