import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI
from typing import List, Optional
from fastapi import Query
from pymongo import MongoClient
import uvicorn
from dotenv import load_dotenv
import os

from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

""""
decir  las palabras magicas:

python main.py

o

python api_mapas/main.py

DENTRO DE LA CARPETA  en local para que funcione
"
"""


load_dotenv()

DB_URL = os.getenv("DB_URL")


app = FastAPI(title="API MAPAS")


@asynccontextmanager
async def lifespan(app: FastAPI): # lo de acerlo asincrono lo saque del ejemplo que tienen los de mongo db pero se supone que ahora lo quieren de esta forma. Tiene coña que que al final lo que haga sea de forma directa mas abajo
    # Startup: create MongoDB client
    app.mongodb_client = AsyncIOMotorClient(DB_URL)
    app.mongodb = app.mongodb_client["Prueba1"]
    print("MongoDB connected")
    
    yield  # FastAPI runs here
    
    # Shutdown: close MongoDB client
    app.mongodb_client.close()
    print("MongoDB connection closed")

@app.get("/")
def root():
    return "API en funcionamiento, /docs, /mapakwh, /mapagas, /mapagas_concreto"

async def leer_ev():
    """Lee la coleccion electricos lo pasa a un df limpio, revisar README.md"""
    client = MongoClient(DB_URL)
    db = client["Prueba1"]
    collection = db["electrico"]
    docs=collection.find({})
    data_ev = pd.DataFrame(docs)


    estacion_df = pd.json_normalize(data_ev["estacion"])
    data_ev = data_ev.drop(columns=["estacion"]).join(estacion_df.add_prefix("estacion."))
    df_exploded = data_ev.explode("lineas").reset_index(drop=True)
    # Normaliza la columna "lineas" (dict → columnas)
    lineas_normalizadas = pd.json_normalize(df_exploded["lineas"])
    # Une las nuevas columnas con el dataset original
    data_ev = df_exploded.drop(columns=["lineas"]).join(lineas_normalizadas.add_prefix("lineas."))
    data_ev['fechaHora'] = pd.to_datetime(data_ev['fechaEmision'].astype(str) + ' ' + data_ev['horaEmision'].astype(str))
    data_ev.drop(['fechaEmision', 'horaEmision'], axis=1, inplace=True)
    return data_ev


async def leer_gas():
    """Lee la coleccion de gasolina lo pasa a un df limpio, revisar README.md"""
    client = MongoClient(DB_URL)
    db = client["Prueba1"]
    collection = db["Combustible"]
    docs=collection.find({})
    data_gas = pd.DataFrame(docs)
    

    estacion_df = pd.json_normalize(data_gas["estacion"])
    data_gas = data_gas.drop(columns=["estacion"]).join(estacion_df.add_prefix("estacion."))
    df_exploded = data_gas.explode("lineas").reset_index(drop=True)
    # Normaliza la columna "lineas" (dict → columnas)
    lineas_normalizadas = pd.json_normalize(df_exploded["lineas"])
    # Une las nuevas columnas con el dataset original
    data_gas = df_exploded.drop(columns=["lineas"]).join(lineas_normalizadas.add_prefix("lineas."))
    data_gas['fechaHora'] = pd.to_datetime(data_gas['fechaEmision'].astype(str) + ' ' + data_gas['horaEmision'].astype(str))
    data_gas.drop(['fechaEmision', 'horaEmision'], axis=1, inplace=True)

    return data_gas

async def leer_peaje():
    """Lee la coleccion de peajes y pasa a un df limpio, revisar README.md"""
    client = MongoClient(DB_URL)
    db = client["Prueba1"]
    collection = db["Peaje"]
    docs=collection.find({})
    data_peaje = pd.DataFrame(docs)
    

    localizacion_df = pd.json_normalize(data_peaje["localizacion"])
    data_peaje["fechaHora"]=pd.to_datetime(data_peaje["fechaHora"])
    data_peaje["importe"] = (data_peaje["importe"].astype(str)  .str.replace(",", ".", regex=False).str.replace("$", "", regex=False).str.replace("€", "", regex=False).astype(float)  )
    data_peaje = data_peaje.drop(columns=["localizacion"]).join(localizacion_df.add_prefix("localizacion."))

    return data_peaje


@app.get("/mapakwh")
async def mapakwh():
    """Devuelve la localizacion de donde meter el heatmap kwh medio en formato json
    Formato: [[lat, lon, precio_medio], ...]
    """
    data_ev = await leer_ev()
    df_map = data_ev.groupby(['estacion.lat', 'estacion.lon'])['lineas.kwh'].mean().reset_index()


    # Crear mapa centrado (ejemplo: centro de España)
    

    # Preparar datos para el HeatMap (lat, lon, peso)
    heat_data = [[row['estacion.lat'], row['estacion.lon'], row['lineas.kwh']] for index, row in df_map.iterrows()]

    # Añadir capa HeatMap
    
    return heat_data

@app.get("/mapagas")
async def mapagas():
    """Devuelve la localización de donde meter el heatmap precio de combustible medio (media de todos) en formato json
    Formato: [[lat, lon, precio_medio], ...]"""
    data_gas =  await leer_gas()
    df_map = data_gas.groupby(['estacion.lat', 'estacion.lon'])['lineas.precioUnitario'].mean().reset_index()


    # Crear mapa centrado (ejemplo: centro de España)
    

    # Preparar datos para el HeatMap (lat, lon, peso)
    heat_data = [[row['estacion.lat'], row['estacion.lon'], row['lineas.precioUnitario']] for index, row in df_map.iterrows()]

    # Añadir capa HeatMap
    
    return heat_data




@app.get("/mapagas_concreto")
async def mapagas_concreto(combustible: Optional[List[str]] = Query(None)):
    """
    Devuelve la localización y el precio unitario medio por estación,
    filtrado opcionalmente por tipo de combustible. Pero no hay diferencia entre coordenadas

    Formato: [[lat, lon, precio_medio], ...]
    """
    data_gas = await leer_gas()

    if combustible:
        data_gas = data_gas[data_gas["lineas.producto"].isin(combustible)]

    df_map = (
        data_gas.groupby(['estacion.lat', 'estacion.lon'])['lineas.precioUnitario']
        .mean()
        .reset_index()
    )

    heat_data = [
        [row['estacion.lat'], row['estacion.lon'], row['lineas.precioUnitario']]
        for _, row in df_map.iterrows()
    ]

    return heat_data


# lo de abajo no tengo claro lo que hace creo que expone el puerto 9000
# el puerto 6000 me da problemas, algo de que "No esta permitido" y "Que podria ser inseguro"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000   # recarga automática en desarrollo
    )

