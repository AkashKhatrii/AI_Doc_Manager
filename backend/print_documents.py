from database import SessionLocal, Document

db = SessionLocal()

documents = db.query(Document).all()

for doc in documents:
    print(f"ID: {doc.id}, Filename: {doc.filename}, Category: {doc.category}")
    print(f"Text preview: {doc.text_content[:2000]}")
    print("-" * 80)

db.close()