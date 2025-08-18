#!/usr/bin/env python3
"""
Quick test script to check your ChromaDB collection
"""

import chromadb
from chromadb.config import Settings

def test_collection():
    try:
        # Connect to ChromaDB
        client = chromadb.HttpClient(
            host="localhost",
            port=8002,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Get collection
        collection = client.get_collection("r_language_reference")
        
        # Check document count
        count = collection.count()
        print(f"Collection contains {count} documents")
        
        if count == 0:
            print("❌ No documents found! You need to run ingest.py first.")
            return
        
        # Test a simple search
        print("\nTesting search with 'machine learning'...")
        results = collection.query(
            query_texts=["machine learning"],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"✅ Found {len(results['documents'][0])} results")
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i]
                source = results['metadatas'][0][i].get('source', 'Unknown')
                print(f"Result {i+1}: {source} (distance: {distance:.3f})")
                print(f"Text: {doc[:200]}...")
        else:
            print("❌ No search results found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_collection()
