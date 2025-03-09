from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import pdfplumber

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methos=["*"],
    allow_headers=["*"],
)

@app.post("/extract-text")
async def exytact_text(file: UploadFile = File(...)):
    content = await file.read()

    if file.content_type == "application/pdf":
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
    elif file.content_type.startswith("image/"):
        image = Image.open(io.BytesIO(content))
        text = pytesseract.image_to_string(image)
    else:
        return {"error": "Unsupported file type"}
    return {"text": text}

