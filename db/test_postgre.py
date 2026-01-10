import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "ai_text_feedback"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432"))
)

cur = conn.cursor()
cur.execute("SELECT 1;")
print(cur.fetchone())

cur.close()
conn.close()
