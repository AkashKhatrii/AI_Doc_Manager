from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import io
import pdfplumber
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA 
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sqlalchemy.orm import Session
from sklearn.metrics.pairwise import cosine_similarity

from database import Document, get_db, SessionLocal
import subprocess
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
vector_db =  {} # Dictionary to store FAISS index per category

def load_existing_vectors():
    global vector_db
    db = SessionLocal()

    docs = db.query(Document).filter(Document.vectorized == True).all()


    category_texts = {}
    for doc in docs:
        if doc.category not in category_texts:
            category_texts[doc.category] = []
        category_texts[doc.category].append(doc.text_content)

    for category, texts in category_texts.items():
        vector_db[category] = FAISS.from_texts(texts, embeddings)

    db.close()

load_existing_vectors()


def extract_text(file_bytes, content_type):
    try:
        if content_type == "application/pdf":
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

        elif content_type.startswith("image/"):
            image = Image.open(io.BytesIO(file_bytes))

            # Convert image to grayscale (improves OCR accuracy)
            image = image.convert("L")

            # Explicitly set tesseract path
            TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

            if not os.path.exists(TESSERACT_PATH):
                raise FileNotFoundError(f"Tesseract not found at {TESSERACT_PATH}. Please install it.")

            # Convert image to a format Tesseract can read (save to BytesIO)
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")  # PNG works well for OCR
            image_bytes.seek(0)

            extracted_text = pytesseract.image_to_string(image)

            if not extracted_text.strip():
                return "No readable text found in the image."

            return extracted_text

        else:
            return None

    except subprocess.SubprocessError as e:
        print(f"Subprocess error in Tesseract: {str(e)}")
        return "Error: Unable to run Tesseract due to system permissions."

    except Exception as e:
        print(f"Error processing file ({content_type}): {str(e)}")
        return f"Error processing file: {str(e)}"

@app.post("/upload")
async def upload_text(file: UploadFile = File(...), category: str = Form(...), db: Session = Depends(get_db)):

    global vector_db

    existing_doc = db.query(Document).filter(Document.filename == file.filename).first()

    if existing_doc and existing_doc.vectorized:
        return {"message": "File already processed", "category": existing_doc.category}
    
    content = await file.read()
    text = extract_text(content, file.content_type)

    if not text:
        return {"error": "could not extract text"}
    
    # Store document metadata in DB
    new_doc = Document(filename=file.filename, category=category, text_content=text, vectorized=True)
    db.add(new_doc)
    db.commit()
    
    # split text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)
    if len(text_chunks) == 0:
        return {"error": "No valid text found to process"}

    # Store extracted text as embeddings in FAISS (category-wise)
    if category not in vector_db:
        vector_db[category] = FAISS.from_texts(text_chunks, embeddings)
    else:
        vector_db[category].add_texts(text_chunks)

    print("stored in vector db:", vector_db[category])

    return {"message": "Document processed successfully", "category": category, "chunks added": len(text_chunks)}


# Predefined categories & descriptions
CATEGORY_DESCRIPTIONS = {
    "medical": "Health records, prescriptions, diagnosis reports, medical history",
    # "finance": "Investment details, salary statements, budget plans, bank transactions",
    "academic": "Exam results, grades, GPA, coursework, academic performance, jobs",
    "legal": "Contracts, agreements, property documents, legal papers",
    "personal": "Diary entries, personal notes, memories, family records, identity cards"
}


# Function to determine query Category using embeddings & cosine similarity
def get_category_from_query(query):
    query_vector = embeddings.embed_query(query)
    category_vectors = {category: embeddings.embed_query(desc) for category, desc in CATEGORY_DESCRIPTIONS.items()}

    # Compute cosine similarity
    similarities = {category: cosine_similarity([query_vector], [vector])[0][0] for category, vector in category_vectors.items()}

    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]

    if best_score < 0.4: # confidence threshold
        return None # uncertain classification
    print(best_category)
    return best_category


# AI search API (Filters by Category before retrieval)
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    global vector_db
    category = get_category_from_query(query)
   

    if not category or category not in vector_db:
        print("error")
        return {"error": "No relevant documents found. Try specifying a category"}
    
    try:
        retriever = vector_db[category].as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query)
        if not retrieved_docs:
            return {"error": "No relevant text was found in the documents. Try rephrasing your query.", "category_used": category}

        # qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

        # response = qa_chain.run(query)
        # print(response)
        # return {"answer": response, "category_used": category}

        document_context = "\n\n".join([doc.page_content for doc in retrieved_docs[:3]])  # Limit to 3 docs for token efficiency

        # prompt to force GPT to use retrieved documents
        prompt = f"""
        You are an AI assistant that can provide insights based on both:
        1. The user's personal documents (extracted from their uploads).
        2. Your general AI knowledge.

        **User's Document Category:** {category}

        **Extracted Document Context:** 
        {document_context}

        **User Question:** {query}

        **Instructions for Answering:**
        - Use the extracted document context where relevant.
        - If the documents don't fully answer the question, provide insights based on general AI knowledge.
        - Ensure the response is relevant to the category: {category}.
        - Structure the response clearly and concisely.
        """

        llm = ChatOpenAI()
        response_message = llm.invoke(prompt) 

        response_text = response_message.content

        if not response_text or response_text.strip() == "":
            return {"error": "AI did not generate a response. Try rephrasing your question.", "category_used": category}

        # print(f"AI Response: {response_text}")
        return {"answer": response_text, "category_used": category}
    except Exception as e:
        print("exception error")
        return {"error": f"Failed to retrieve answer: {str(e)}"}