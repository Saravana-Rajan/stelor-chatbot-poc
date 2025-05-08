import pdfplumber
import pandas as pd
import fitz  # PyMuPDF
import docx
from PIL import Image
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from django.conf import settings
import os
import mimetypes
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Initialize sentence transformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="document_chunks")

def detect_file_type(file_path: str) -> str:
    """Detect file type using file extension and mime types."""
    mime_type, _ = mimetypes.guess_type(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    if mime_type == 'application/pdf' or extension == '.pdf':
        return 'pdf'
    elif mime_type == 'text/csv' or extension == '.csv':
        return 'csv'
    elif mime_type == 'text/plain' or extension == '.txt':
        return 'txt'
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or extension == '.docx':
        return 'docx'
    elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or extension == '.xlsx':
        return 'xlsx'
    elif mime_type and mime_type.startswith('image/'):
        return 'image'
    else:
        raise ValueError(f"Unsupported file type: {mime_type or extension}")

def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process PDF file and return chunks with page numbers."""
    chunks = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                chunks.append({
                    'content': text,
                    'page_number': page_num,
                    'metadata': {
                        'file_type': 'pdf',
                        'page': page_num
                    }
                })
    
    return chunks

def process_csv(file_path: str) -> List[Dict[str, Any]]:
    """Process CSV file and return chunks with table information."""
    df = pd.read_csv(file_path)
    chunks = []
    
    # Convert DataFrame to string representation
    table_str = df.to_string()
    chunks.append({
        'content': table_str,
        'metadata': {
            'file_type': 'csv',
            'columns': df.columns.tolist(),
            'rows': len(df)
        }
    })
    
    return chunks

def process_txt(file_path: str) -> List[Dict[str, Any]]:
    """Process text file and return chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into paragraphs
    paragraphs = content.split('\n\n')
    chunks = []
    
    for i, para in enumerate(paragraphs):
        if para.strip():
            chunks.append({
                'content': para.strip(),
                'metadata': {
                    'file_type': 'txt',
                    'paragraph': i + 1
                }
            })
    
    return chunks

def process_docx(file_path: str) -> List[Dict[str, Any]]:
    """Process Word document and return chunks."""
    doc = docx.Document(file_path)
    chunks = []
    
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            chunks.append({
                'content': para.text.strip(),
                'metadata': {
                    'file_type': 'docx',
                    'paragraph': i + 1
                }
            })
    
    return chunks

def process_image(file_path: str) -> List[Dict[str, Any]]:
    """Process image file using Gemini Vision."""
    image = Image.open(file_path)
    
    # Use Gemini Vision to analyze the image
    response = model.generate_content([
        "Analyze this image and provide a detailed description of its contents.",
        image
    ])
    
    chunks = [{
        'content': response.text,
        'metadata': {
            'file_type': 'image',
            'format': image.format,
            'size': image.size
        }
    }]
    
    return chunks

def get_embeddings(text: str) -> List[float]:
    """Generate embeddings for text using sentence-transformers."""
    return embedding_model.encode(text).tolist()

def store_chunks(chunks: List[Dict[str, Any]], document_id: str):
    """Store chunks in ChromaDB with embeddings."""
    for i, chunk in enumerate(chunks):
        embedding = get_embeddings(chunk['content'])
        collection.add(
            documents=[chunk['content']],
            embeddings=[embedding],
            metadatas=[{
                'document_id': document_id,
                'chunk_index': i,
                **chunk['metadata']
            }]
        )

def query_chunks(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Query ChromaDB for relevant chunks."""
    query_embedding = get_embeddings(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return [
        {
            'content': doc,
            'metadata': meta
        }
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ]

def generate_response(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Generate response using Gemini with context."""
    context = "\n\n".join([
        f"Source: {chunk['metadata'].get('file_type', 'unknown')} - "
        f"{chunk['metadata'].get('page', chunk['metadata'].get('paragraph', 'N/A'))}\n"
        f"Content: {chunk['content']}"
        for chunk in context_chunks
    ])
    
    prompt = f"""Answer the user's question using ONLY the provided context.
Include file names and pages/paragraphs when citing sources.

Context:
{context}

Question: {query}

Answer:"""
    
    response = model.generate_content(prompt)
    return response.text 