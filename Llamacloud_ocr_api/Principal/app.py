from fastapi import FastAPI, UploadFile, File
from llama_cloud_services import LlamaExtract
from llama_cloud_services import SourceText

from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="OCR")

api_key = os.getenv("LLAMA-API-KEY")
agent_name_gas = os.getenv("NOMBRE-AGENTE-GASOLINERA")
agent_name_peaje = os.getenv("NOMBRE-AGENTE_PEAJE")
extractor = LlamaExtract(api_key=api_key)
agent_gas = extractor.get_agent(name=agent_name_gas)
agent_peaje = extractor.get_agent(name=agent_name_peaje)

@app.post("/gasolineras/")
async def extract_data(file: UploadFile = File(...)):

    file_bytes = await file.read()
    filename = file.filename
    # Si agent.extract acepta bytes:
    result = agent_gas.extract(SourceText(file=file_bytes, filename=filename))
    #(file_bytes)
    return {"data": result.data}



@app.post("/peaje/")
async def extract_data(file: UploadFile = File(...)):

    file_bytes = await file.read()
    filename = file.filename
    # Si agent.extract acepta bytes:
    result = agent_peaje.extract(SourceText(file=file_bytes, filename=filename))
    #(file_bytes)
    return {"data": result.data}