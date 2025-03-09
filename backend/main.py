from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import pdfplumber
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
pytesseract.pytesseract.tesseract_cmd =  os.getenv("PATH_TO_TESSERACT")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = OpenAIEmbeddings()
vector_db =  None

def extract_text(file_bytes, content_type):
    if content_type == "application/pdf":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif content_type.startswith("image/"):
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image)
    
    return None

@app.post("/upload")
async def upload_text(file: UploadFile = File(...)):

    global vector_db

    content = await file.read()
    text = extract_text(content, file.content_type)

    if not text:
        return {"error": "could not extract text"}
    
    # split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)
    print(text)
    if len(text_chunks) == 0:
        return {"error": "No valid text found to process"}

    # Store extracted text as embeddings in FAISS
    vector_db = FAISS.from_texts(text_chunks, embeddings)  

    return {"message": "Document processed successfully", "chunks added": len(text_chunks)}

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    global vector_db  # Ensure we use the latest FAISS instance

    if vector_db is None:
        return {"error": "No documents uploaded yet. Please upload a document first."}

    try:
        retriever = vector_db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

        response = qa_chain.run(query)
        return {"answer": response}
    except Exception as e:
        return {"error": f"Failed to retrieve answer: {str(e)}"}