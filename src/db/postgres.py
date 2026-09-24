import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set")

    return psycopg2.connect(DATABASE_URL)