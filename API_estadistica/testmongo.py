# testmongo_atlas.py
from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd

# Cargar variables de entorno
load_dotenv()

DB_URL = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME", "Prueba1")

# Conectar a MongoDB Atlas
try:
    client = MongoClient(DB_URL)
    db = client[DB_NAME]
    print(f"Conectado a la base de datos: {DB_NAME}")
except Exception as e:
    print("Error conectando a MongoDB:", e)
    exit(1)

# Colecciones a comprobar
collections = ["Combustible", "electricos", "Peaje"]

for col_name in collections:
    print("\n" + "="*50)
    print(f"Colección: {col_name}")
    coll = db[col_name]
    try:
        total_docs = coll.count_documents({})
        print(f"Total documentos: {total_docs}")
        docs = list(coll.find({}).limit(5))
        df = pd.DataFrame(docs)
        print("Columnas detectadas:", df.columns.tolist())
        print("Primeros 5 documentos:")
        print(df.head())
    except Exception as e:
        print(f"Error accediendo a la colección {col_name}:", e)
