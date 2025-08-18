#!/usr/bin/env python3
"""
RAG Chatbot Application - Minimal Version
Uses ChromaDB for document retrieval and Ollama for text generation
"""

import gradio as gr
import chromadb
from chromadb.config import Settings
import requests
import json
import time
from typing import List, Dict, Any

class RAGChatbot:
    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8002, 
                 ollama_host: str = "127.0.0.1", ollama_port: int = 11434):
        """Initialize the RAG chatbot"""
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
        except:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        self.collection_name = "r_language_reference"
        self.collection = self._get_collection()
        
    def _get_collection(self):
        """Get the ChromaDB collection"""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            print(f"Connected to collection: {self.collection_name}")
            return collection
        except Exception as e:
            print(f"Error connecting to collection: {e}")
            return None
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in ChromaDB"""
        if not self.collection:
            return []
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            return documents
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            print(f"Checking available models at {self.ollama_url}/api/tags")
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Response data: {data}")
                models = data.get('models', [])
                model_names = [model['name'] for model in models]
                print(f"Found models: {model_names}")
                return model_names
            else:
                print(f"Error response: {response.text}")
                return []
        except Exception as e:
            print(f"Exception checking models: {e}")
            return []
    
    def call_ollama(self, prompt: str, model: str = "llama2") -> str:
        """Call Ollama API to generate response"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            print(f"Calling Ollama with model {model} (no timeout)...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=None  # No timeout - wait as long as needed
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', 'No response generated')
                print(f"Ollama response completed in {elapsed_time:.2f} seconds")
                return f"[{elapsed_time:.2f}s] {response_text}"
            else:
                return f"Error: HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please make sure Ollama is running on 127.0.0.1:11434"
        except Exception as e:
            return f"Error calling Ollama: {e}"
    
    def create_rag_prompt(self, user_question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Create a RAG prompt with context from relevant documents"""
        if not relevant_docs:
            return user_question
        
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            context_parts.append(f"Document {i} (from {source}):\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Create the RAG prompt
        prompt = f"""You are a helpful assistant with access to R programming language documentation. 
Use the following context to answer the user's question. If the context doesn't contain relevant information, 
say so and provide a general helpful response about R programming.

Context:
{context}

User Question: {user_question}

Please provide a clear, helpful answer based on the context provided."""
        
        return prompt

def respond(message, history, model):
    """Main chat function for Gradio"""
    overall_start = time.time()
    print(f"\n=== NEW QUERY ===")
    print(f"Query: {message}")
    print(f"Model: {model}")
    
    if not message.strip():
        return ""
    
    # Create chatbot instance for this request
    chatbot = RAGChatbot()
    
    # Check available models first
    available_models = chatbot.get_available_models()
    print(f"Available models: {available_models}")
    
    if model not in available_models:
        return f"Error: Model '{model}' is not available. Available models: {', '.join(available_models)}"
    
    # Search for relevant documents
    search_start = time.time()
    relevant_docs = chatbot.search_documents(message, n_results=3)
    search_time = time.time() - search_start
    print(f"ChromaDB search: {search_time:.2f}s, found {len(relevant_docs)} documents")
    
    # Create RAG prompt
    prompt_start = time.time()
    rag_prompt = chatbot.create_rag_prompt(message, relevant_docs)
    prompt_time = time.time() - prompt_start
    print(f"Prompt creation: {prompt_time:.2f}s")
    
    # Get response from Ollama
    print("Calling Ollama...")
    response = chatbot.call_ollama(rag_prompt, model)
    
    # Add source information if documents were found
    if relevant_docs:
        sources = [doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]
        source_info = f"\n\nSources: {', '.join(set(sources))}"
        response += source_info
    
    # Calculate total time
    total_time = time.time() - overall_start
    print(f"Total query time: {total_time:.2f}s")
    print("=== END QUERY ===\n")
    
    # Ensure response is a clean string and return it directly
    final_response = str(response).strip()
    return final_response

# Create the interface using the simplest possible components
iface = gr.ChatInterface(
    fn=respond,
    title="R Language RAG Chatbot - Research Mode",
    description="Research tool for testing different models with R programming questions. Response times are shown in brackets. No timeouts - will wait for complete responses.",
    examples=[
        ["What is a vector in R?"],
        ["How do I create a data frame?"],
        ["What are the basic data types in R?"]
    ],
    additional_inputs=[
        gr.Dropdown(
            choices=["gemma3:270m", "gemma:latest", "qwen3:latest", "yi-coder:latest", "gpt-oss:20b"],
            value="gemma3:270m",
            label="Model"
        )
    ]
)

if __name__ == "__main__":
    iface.launch(
        server_name="localhost",
        server_port=7860,  # Changed port to avoid conflicts
        share=False,
        show_error=True,
        quiet=False
    )
