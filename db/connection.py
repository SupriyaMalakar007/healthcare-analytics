
"""
Shared database connection layer.

Reads connection information from environment variables
(see .env.example).

This connection uses SQLAlchemy's URL.create() so special
characters in the PostgreSQL password, such as @, are handled
correctly.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker


# Load variables from .env
load_dotenv()


# Database configuration
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "healthcare_analytics")


# Build the database URL safely.
# URL.create() prevents special characters such as @ in the
# password from being interpreted as part of the hostname.
DATABASE_URL = URL.create(
    "postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)


# Create SQLAlchemy engine
# pool_pre_ping helps recover stale connections.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# Create database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_engine():
    """Return the SQLAlchemy database engine."""
    return engine


def get_session():
    """Return a new database session."""
    return SessionLocal()


def test_connection() -> bool:
    """
    Test whether the application can connect to PostgreSQL.

    Returns:
        True  - connection successful
        False - connection failed
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return True

    except Exception as e:
        print(f"[db] connection failed: {e}")
        return False


if __name__ == "__main__":
    ok = test_connection()

    if ok:
        print("Connected ✅")
    else:
        print("Connection failed ❌")
