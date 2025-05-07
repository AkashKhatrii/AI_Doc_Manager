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

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
async def upload_text(files: list[UploadFile] = File(...), category: str = Form(...), db: Session = Depends(get_db)):

    global vector_db
    all_chunks = []
    for file in files:
        existing_doc = db.query(Document).filter(Document.filename == file.filename).first()

        if existing_doc and existing_doc.vectorized:
            continue

    # if existing_doc and existing_doc.vectorized:
    #     return {"message": "File already processed", "category": existing_doc.category}

        content = await file.read()
        text = extract_text(content, file.content_type)
        if not text:
            continue

        doc = Document(filename=file.filename, category=category, text_content=text, vectorized=True)
        db.add(doc)    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        text_chunks = text_splitter.split_text(text)
        clean_chunks = [c for c in text_chunks if isinstance(c, str) and c.strip()]

        all_chunks.extend(clean_chunks)
    
    db.commit()
    if not all_chunks:
        return {"error": "No valid text chunks found across all files."}

    try:
        if category not in vector_db:
            vector_db[category] = FAISS.from_texts(all_chunks, embeddings)
        else:
            vector_db[category].add_texts(all_chunks)
    except Exception as e:
        print("FAISS error:", str(e))
        return {"error": f"Failed to embed chunks: {str(e)}"}

    return {
        "message": f"{len(files)} file(s) processed successfully.",
        "category": category,
        "chunks_added": len(all_chunks)
    }


# Predefined categories & descriptions
CATEGORY_DESCRIPTIONS = {
    "medical": (
        "Health records, doctor prescriptions, medical reports, diagnosis documents, "
        "lab test results, hospital bills, vaccination history, health insurance papers, "
        "radiology scans, surgery notes, clinical notes, prescriptions."
    ),
    "academic": (
        "Educational documents, exam results, report cards, transcripts, GPA reports, "
        "certificates, project submissions, coursework, degree completion, academic papers, "
        "student ID, resume for internships or jobs, recommendation letters."
    ),
    "legal": (
        "Legal contracts, court documents, affidavits, property ownership papers, legal agreements, "
        "government-issued documents, rental agreements, licenses, power of attorney, notary documents, "
        "wills, legal notices, deeds, judicial records."
    ),
    "personal": (
        "Personal notes, family letters, identity cards (passport, Aadhar, PAN), birthday invitations, "
        "travel plans, insurance documents, diaries, photographs with captions, memories, family trees, "
        "journal entries, personal resumes, handwritten notes, life planning documents."
    )
}


# Function to determine query Category using embeddings & cosine similarity
def get_category_from_query(query):
    query_vector = embeddings.embed_query(query)
    category_vectors = {category: embeddings.embed_query(desc) for category, desc in CATEGORY_DESCRIPTIONS.items()}

    # Compute cosine similarity
    similarities = {category: cosine_similarity([query_vector], [vector])[0][0] for category, vector in category_vectors.items()}
    print(similarities)
    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]

    # if best_score < 0.4: # confidence threshold
    #     return None # uncertain classification
    print(best_category)
    return best_category


# AI search API (Filters by Category before retrieval)
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    global vector_db
    category = get_category_from_query(query)
    print("category", category)
   

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

        document_context = "\n\n".join([doc.page_content for doc in retrieved_docs])  # Limit to 3 docs for token efficiency

        # prompt to force GPT to use retrieved documents
        prompt = f"""
            You are a helpful assistant. A user has uploaded documents under the category "{category}" and asked a question.

            Your job is to:
            - Use the document context below to answer the question directly.
            - If the documents do not contain the full answer, supplement with general knowledge but make it clear.
            - Keep the response factual, concise, and well-structured.

            DOCUMENT CONTEXT:
            {document_context}

            QUESTION:
            {query}

            ANSWER:
            """


        llm = ChatOpenAI(model="gpt-4-turbo")
        response_message = llm.invoke(prompt) 

        response_text = response_message.content

        if not response_text or response_text.strip() == "":
            return {"error": "AI did not generate a response. Try rephrasing your question.", "category_used": category}

        # print(f"AI Response: {response_text}")
        return {"answer": response_text, "category_used": category}
    except Exception as e:
        print("exception error")
        return {"error": f"Failed to retrieve answer: {str(e)}"}