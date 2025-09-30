import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from doctr.io import DocumentFile # Convierte PDF e imágenes en objetos que docTR puede leer
from doctr.models import ocr_predictor # Carga el modelo OCR


app = FastAPI(
    title = "OCR Tickets API",
    description = "OCR con docTR para procesar tickets (PDF/imagen)",
    version = "1.0.0",
)


ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True) # Modelo de OCR


# Función auxiliar para convertir el archivo de entrada en 
# un DocumentFile que sea leido por el OCR y devuelva un JSON estructurado


def process_document(file_bytes: bytes, file_type :str):
    if file_type == "application/pdf":
        doc = DocumentFile.from_pdf(io.BytesIO(file_bytes))
    elif file_type in ["image/jpeg", "image/png"]:
        doc = DocumentFile.from_images(io.BytesIO(file_bytes))
    else:
        raise HTTPException(status_code = 400, detail="Tipo de archivo no soportado") # Arroja error

    result = ocr_model(doc)
    return result.export()



# Construcción del End point:

@app.post("/ocr")
async def upload_invoice(file: UploadFile = File(...)):
    file_bytes = await file.read()

    try:
        json_output = process_document(file_bytes, file.content_type)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error en OCR: {e}')
    

    return json_output