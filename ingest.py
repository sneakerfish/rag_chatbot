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
import unicodedata
import chromadb
from chromadb.config import Settings
import uuid

try:
    import pymupdf  # Much better word spacing than PyPDF2 and about 10x faster
except ImportError:  # pragma: no cover
    pymupdf = None
import PyPDF2

from chunking import default_embedding_function, fixed_chunk_text, semantic_chunk_text

class PDFIngester:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8002,
                 chunking: str = "semantic", chunk_size: int = 1000, overlap: int = 200,
                 breakpoint_percentile: float = 90.0, max_chunk_size: int = 1500,
                 min_chunk_size: int = 200, collection_name: str = "r_language_reference"):
        """Initialize the PDF ingester with ChromaDB connection and chunking options

        chunking: "semantic" (default) places chunk boundaries where the topic
        changes, so an explanation is not cut in half. "fixed" is the original
        sliding window of chunk_size characters with overlap.
        """
        if chunking not in ("semantic", "fixed"):
            raise ValueError(f"Unknown chunking strategy: {chunking}")
        self.chunking = chunking
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        # Lazily created so "fixed" mode never loads the embedding model.
        self._embed = None
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
        self.collection_name = collection_name
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
        """Extract text content from a PDF file.

        PyMuPDF is used when available: PyPDF2 runs words together on many
        PDFs (e.g. anything built by Sphinx/LaTeX), which ruins both retrieval
        and sentence splitting. Ligatures such as "fi" are folded to plain
        letters so they match query text.
        """
        try:
            if pymupdf is not None:
                with pymupdf.open(pdf_path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
            else:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = "\n".join((page.extract_text() or "") for page in pdf_reader.pages)
            return unicodedata.normalize("NFKC", text)
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using the configured strategy"""
        if self.chunking == "fixed":
            return fixed_chunk_text(text, chunk_size=self.chunk_size, overlap=self.overlap)

        if self._embed is None:
            # Same model ChromaDB uses to embed the chunks for retrieval.
            self._embed = default_embedding_function()
        return semantic_chunk_text(
            text,
            self._embed,
            breakpoint_percentile=self.breakpoint_percentile,
            max_chunk_size=self.max_chunk_size,
            min_chunk_size=self.min_chunk_size,
        )
    
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
                    'file_path': pdf_path,
                    'chunking': self.chunking
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
            
            # Add to ChromaDB collection in batches; a single add is capped
            # at a few thousand records by the server.
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                self.collection.add(
                    ids=ids[i:i + batch_size],
                    documents=texts[i:i + batch_size],
                    metadatas=metadatas[i:i + batch_size]
                )
                print(f"  Stored {min(i + batch_size, len(ids))}/{len(ids)} chunks")
            
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
    parser.add_argument("--collection", default="r_language_reference",
                        help="ChromaDB collection to store chunks in (default: r_language_reference)")
    parser.add_argument("--chunking", choices=["semantic", "fixed"], default="semantic",
                        help="Chunking strategy: 'semantic' breaks where the topic changes, "
                             "'fixed' is a sliding character window (default: semantic)")
    semantic = parser.add_argument_group("semantic chunking options")
    semantic.add_argument("--breakpoint-percentile", type=float, default=90.0,
                          help="Distance percentile above which a topic shift becomes a chunk "
                               "boundary; lower means more, smaller chunks (default: 90)")
    semantic.add_argument("--max-chunk-size", type=int, default=1500,
                          help="Split chunks longer than this many characters (default: 1500)")
    semantic.add_argument("--min-chunk-size", type=int, default=200,
                          help="Merge chunks shorter than this many characters into a neighbour "
                               "(default: 200)")
    fixed = parser.add_argument_group("fixed chunking options")
    fixed.add_argument("--chunk-size", type=int, default=1000,
                       help="Characters per chunk (default: 1000)")
    fixed.add_argument("--overlap", type=int, default=200,
                       help="Characters shared between consecutive chunks (default: 200)")
    
    args = parser.parse_args()
    
    try:
        ingester = PDFIngester(
            chroma_host=args.host,
            chroma_port=args.port,
            collection_name=args.collection,
            chunking=args.chunking,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            breakpoint_percentile=args.breakpoint_percentile,
            max_chunk_size=args.max_chunk_size,
            min_chunk_size=args.min_chunk_size,
        )
        ingester.ingest_folder(args.folder)
        ingester.get_collection_info()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
