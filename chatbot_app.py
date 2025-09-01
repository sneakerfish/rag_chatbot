#!/usr/bin/env python3
"""
RAG Chatbot Application - Multi-Collection Version
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
    def __init__(self, collection_name: str, chroma_host: str = "localhost", chroma_port: int = 8002, 
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
        self.collection_name = collection_name
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
    
    def search_documents_diverse(self, query: str, n_results: int = 10, max_per_source: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documents ensuring diversity across sources"""
        if not self.collection:
            return []
        
        try:
            # Get more results initially to allow for filtering
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results to filter from
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results['documents'] and results['documents'][0]:
                source_counts = {}
                
                for i, doc in enumerate(results['documents'][0]):
                    source = results['metadatas'][0][i].get('source', 'Unknown')
                    
                    # Limit documents per source
                    if source_counts.get(source, 0) < max_per_source:
                        documents.append({
                            'text': doc,
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i]
                        })
                        source_counts[source] = source_counts.get(source, 0) + 1
                    
                    # Stop if we have enough diverse results
                    if len(documents) >= n_results:
                        break
                
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
    
    def create_rag_prompt(self, user_question: str, relevant_docs: List[Dict[str, Any]], 
                         collection_name: str, custom_prompt: str = None) -> str:
        """Create a RAG prompt with context from relevant documents"""
        if not relevant_docs:
            return user_question
        
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            context_parts.append(f"Document {i} (from {source}):\n{doc['text']}\n")
        
        context = "\n".join(context_parts)
        
        # Use custom prompt if provided, otherwise use default based on collection
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            # Default prompts based on collection type
            if "r_language" in collection_name.lower():
                prompt_template = """You are a helpful assistant with access to R programming language documentation. 
Use the following context to answer the user's question. If the context doesn't contain relevant information, 
say so and provide a general helpful response about R programming.

Context:
{context}

User Question: {question}

Please provide a clear, helpful answer based on the context provided."""
            elif "python" in collection_name.lower():
                prompt_template = """You are a helpful assistant with access to Python programming documentation. 
Use the following context to answer the user's question. If the context doesn't contain relevant information, 
say so and provide a general helpful response about Python programming.

Context:
{context}

User Question: {question}

Please provide a clear, helpful answer based on the context provided."""
            else:
                prompt_template = """You are a helpful assistant with access to documentation. 
Use the following context to answer the user's question. If the context doesn't contain relevant information, 
say so and provide a general helpful response.

Context:
{context}

User Question: {question}

Please provide a clear, helpful answer based on the context provided."""
        
        # Create the RAG prompt
        prompt = prompt_template.format(context=context, question=user_question)
        return prompt

def get_available_collections(chroma_host: str = "localhost", chroma_port: int = 8002) -> List[str]:
    """Get list of available ChromaDB collections"""
    try:
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"Available collections: {collection_names}")
        return collection_names
    except Exception as e:
        print(f"Error getting collections: {e}")
        return ["r_language_reference"]  # Fallback to default

def respond(message, history, collection_name, model, custom_prompt):
    """Main chat function for Gradio"""
    overall_start = time.time()
    print(f"\n=== NEW QUERY ===")
    print(f"Query: {message}")
    print(f"Collection: {collection_name}")
    print(f"Model: {model}")
    
    if not message.strip():
        return ""
    
    if not collection_name:
        return "Error: Please select a collection to use."
    
    # Create chatbot instance for this request
    chatbot = RAGChatbot(collection_name=collection_name)
    
    # Check available models first
    available_models = chatbot.get_available_models()
    print(f"Available models: {available_models}")
    
    if model not in available_models:
        return f"Error: Model '{model}' is not available. Available models: {', '.join(available_models)}"
    
    # Search for relevant documents
    search_start = time.time()
    relevant_docs = chatbot.search_documents_diverse(message, n_results=10, max_per_source=3)
    search_time = time.time() - search_start
    print(f"ChromaDB search: {search_time:.2f}s, found {len(relevant_docs)} documents")
    
    # Create RAG prompt
    prompt_start = time.time()
    rag_prompt = chatbot.create_rag_prompt(
        message, 
        relevant_docs, 
        collection_name, 
        custom_prompt if custom_prompt.strip() else None
    )
    prompt_time = time.time() - prompt_start
    print(f"Prompt creation: {prompt_time:.2f}s")
    
    # Get response from Ollama
    print("Calling Ollama...")
    response = chatbot.call_ollama(rag_prompt, model)
    
    # Add source information if documents were found
    if relevant_docs:
        sources = [doc['metadata'].get('source', 'Unknown') for doc in relevant_docs]
        source_info = f"\n\nSources consulted: {', '.join(sources)}"
        response += source_info
    
    # Calculate total time
    total_time = time.time() - overall_start
    print(f"Total query time: {total_time:.2f}s")
    print("=== END QUERY ===\n")
    
    # Ensure response is a clean string and return it directly
    final_response = str(response).strip()
    return final_response

# Get available collections for the dropdown
available_collections = get_available_collections()

# Create the interface with collection selection and custom prompt
iface = gr.ChatInterface(
    fn=respond,
    title="Multi-Collection RAG Chatbot - Research Mode",
    description="Research tool for testing different models and collections. Select a collection and optionally provide a custom prompt template.",
    additional_inputs=[
        gr.Dropdown(
            choices=available_collections if available_collections else ["r_language_reference"],
            value=available_collections[0] if available_collections else "r_language_reference",
            label="Collection",
            info="Select which document collection to search"
        ),
        gr.Dropdown(
            choices=["gemma3:270m", "gemma:latest", "qwen3:latest", "yi-coder:latest", "gpt-oss:20b"],
            value="gemma3:270m",
            label="Model"
        ),
        gr.Textbox(
            label="Custom Prompt Template (Optional)",
            placeholder="Leave empty to use default prompt. Use {context} and {question} as placeholders.",
            lines=3,
            info="Custom prompt template. Use {context} for document context and {question} for user question."
        )
    ],
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-container {
        min-height: 500px !important;
    }
    .chat-message {
        min-height: 60px !important;
    }
    """
)

if __name__ == "__main__":
    iface.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
