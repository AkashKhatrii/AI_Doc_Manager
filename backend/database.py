from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
import datetime
import os


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./documents.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Document Metadata model
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    category = Column(String, index=True)
    text_content = Column(Text) # full extracted text
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)

# function to get DB session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


