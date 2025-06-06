"""Database connection utilities. Handles database connections and queries."""

import logging
from contextlib import contextmanager

import pandas as pd
import psycopg2

from bonn_thesis.config import config_database

# Set up logger
logger = logging.getLogger(__name__)


@contextmanager
def db_connection():
    """Context manager for database connections.

    Returns a connection that will be automatically closed when the context exits.
    """
    conn = None
    try:
        # Connect using the configuration from config.py
        conn = psycopg2.connect(
            host=config_database["database"]["host"],
            user=config_database["database"]["user"],
            password=config_database["database"]["password"],
            database=config_database["database"]["database"],
            port=config_database["database"]["port"],
        )
        yield conn
    except Exception:
        logger.exception("Database connection error")
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")


def query_to_dataframe(sql, params=None):
    """Execute SQL query and return results as a pandas DataFrame."""
    with db_connection() as conn:
        logger.info("Executing query: %s...", sql[:100])
        return pd.read_sql_query(sql, conn, params=params)


def execute_query(sql, params=None):
    """Execute SQL query without returning results."""
    with db_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params or {})
        conn.commit()
