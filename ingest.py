#!/usr/bin/env python3
"""
PDF Ingest Script for RAG Chatbot
Processes PDF files and stores chunks in ChromaDB collection
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import PyPDF2
import hashlib
import uuid

class PDFIngester:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8002):
        """Initialize the PDF ingester with ChromaDB connection"""
        try:
            # Try basic connection first
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
        except:
            # Fall back to settings if basic connection fails
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
        self.collection_name = "r_language_reference"
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "R Language Reference Documentation"}
            )
            print(f"Created new collection: {self.collection_name}")
        return collection
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def process_pdf_file(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process a single PDF file and return chunks with metadata"""
        print(f"Processing: {pdf_path}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return []
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Create documents with metadata
        documents = []
        filename = Path(pdf_path).name
        
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            documents.append({
                'id': doc_id,
                'text': chunk,
                'metadata': {
                    'source': filename,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'file_path': pdf_path
                }
            })
        
        print(f"  Created {len(documents)} chunks from {filename}")
        return documents
    
    def ingest_folder(self, folder_path: str):
        """Process all PDF files in a folder"""
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Error: Folder {folder_path} does not exist")
            return
        
        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.process_pdf_file(str(pdf_file))
            all_documents.extend(documents)
        
        if all_documents:
            # Prepare data for ChromaDB
            ids = [doc['id'] for doc in all_documents]
            texts = [doc['text'] for doc in all_documents]
            metadatas = [doc['metadata'] for doc in all_documents]
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"Successfully ingested {len(all_documents)} chunks into ChromaDB")
        else:
            print("No documents to ingest")
    
    def get_collection_info(self):
        """Print information about the collection"""
        count = self.collection.count()
        print(f"Collection '{self.collection_name}' contains {count} documents")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF files into ChromaDB for RAG chatbot")
    parser.add_argument("folder", help="Path to folder containing PDF files")
    parser.add_argument("--host", default="localhost", help="ChromaDB host (default: localhost)")
    parser.add_argument("--port", type=int, default=8002, help="ChromaDB port (default: 8002)")
    
    args = parser.parse_args()
    
    try:
        ingester = PDFIngester(chroma_host=args.host, chroma_port=args.port)
        ingester.ingest_folder(args.folder)
        ingester.get_collection_info()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
