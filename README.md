# R Language RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that uses ChromaDB for document storage and Ollama for text generation. This system is designed to answer questions about R programming language based on ingested PDF documentation.

## Features

- **PDF Document Ingestion**: Automatically processes and chunks PDF files
- **Vector Database**: Uses ChromaDB for efficient document retrieval
- **RAG Implementation**: Combines document retrieval with LLM generation
- **Gradio Interface**: Modern web-based chat interface
- **Model Flexibility**: Easy switching between different Ollama models
- **Source Attribution**: Shows which documents were used for answers

## Prerequisites

Before running this system, you need:

1. **ChromaDB Server** running on port 8002
2. **Ollama** running on port 11434 with at least one model installed
3. **Python 3.8+** with pip

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Start ChromaDB Server

Make sure ChromaDB is running on port 8002. You can start it with:

```bash
chroma run --host localhost --port 8002
```

### 2. Start Ollama

Make sure Ollama is running and you have at least one model installed:

```bash
# Start Ollama (if not already running)
ollama serve

# Install a model (example)
ollama pull llama2
```

### 3. Prepare Your PDF Files

Place your R programming documentation PDF files in a folder. The system will process all `.pdf` files in the specified directory.

## Usage

### Step 1: Ingest PDF Documents

Use the ingest script to process your PDF files and store them in ChromaDB:

```bash
python ingest.py /path/to/your/pdf/folder
```

Optional parameters:
- `--host`: ChromaDB host (default: localhost)
- `--port`: ChromaDB port (default: 8002)
- `--chunking`: `semantic` (default) or `fixed`, see [Chunking](#chunking) below

Example:
```bash
python ingest.py ./r_docs --host localhost --port 8002
```

The script will:
- Extract text from all PDF files in the folder
- Split text into chunks at semantic breakpoints (where the topic changes)
- Store chunks in ChromaDB collection "R Language Reference"
- Display progress and final document count

### Step 2: Start the Chatbot Application

Launch the Gradio web interface:

```bash
python chatbot_app.py
```

The application will be available at `http://localhost:7860`

## Using the Chatbot

1. **Open the Web Interface**: Navigate to `http://localhost:7860`
2. **Select a Model**: Choose from available Ollama models in the dropdown
3. **Ask Questions**: Type your R programming questions
4. **Get Answers**: The system will:
   - Search for relevant documents in ChromaDB
   - Create a RAG prompt with context
   - Generate an answer using the selected Ollama model
   - Show source documents used

## Configuration

### Available Models

The system comes pre-configured with common Ollama models:
- `llama2` (default)
- `llama2:7b`
- `llama2:13b`
- `codellama`
- `mistral`
- `neural-chat`

You can modify the model list in `chatbot_app.py` to match your installed models.

### Chunking

How a document is cut into chunks decides what the chatbot can retrieve. A
fixed-size window (the original behaviour) cuts every N characters, which
routinely splits an explanation in half so that neither piece answers the
question on its own.

The default is now **semantic breakpoint chunking** (`chunking.py`):

1. The extracted text is cleaned up (line wrapping and hyphenation removed,
   section headings kept with the section they introduce).
2. It is split into sentences, and each sentence is embedded together with its
   neighbours using the same `all-MiniLM-L6-v2` model ChromaDB uses for search.
3. The cosine distance between each pair of adjacent sentences is computed. A
   large jump means the topic changed, so a chunk boundary is placed there.
4. Chunks over `--max-chunk-size` are split again at their weakest internal
   boundary; chunks under `--min-chunk-size` are merged into a neighbour.

Options for `ingest.py`:

| Flag | Default | Effect |
|------|---------|--------|
| `--chunking semantic\|fixed` | `semantic` | Strategy to use |
| `--breakpoint-percentile` | `90` | Distance percentile that counts as a topic shift. Lower gives more, smaller chunks |
| `--max-chunk-size` | `1500` | Upper bound on chunk length in characters |
| `--min-chunk-size` | `200` | Fragments shorter than this are merged |
| `--chunk-size` | `1000` | Fixed mode only: window size |
| `--overlap` | `200` | Fixed mode only: overlap between windows |

Semantic chunking embeds every sentence once, so ingesting is slower than the
fixed window (roughly 15 seconds for a 100-page manual on a laptop CPU). Each
chunk's metadata records which strategy produced it under the `chunking` key.
If you re-ingest an existing collection with a different strategy, delete the
collection first or you will have both sets of chunks side by side.

To keep the old behaviour:

```bash
python ingest.py ./r_docs --chunking fixed
```

Run the chunking unit tests with:

```bash
python -m pytest test_chunking.py
```

### System Status

The web interface includes real-time status indicators for:
- ChromaDB connection and document count
- Ollama connection
- Collection information

## Troubleshooting

### Common Issues

1. **ChromaDB Connection Error**
   - Ensure ChromaDB server is running on port 8002
   - Check firewall settings
   - Verify the host and port in the scripts

2. **Ollama Connection Error**
   - Ensure Ollama is running (`ollama serve`)
   - Check if the model is installed (`ollama list`)
   - Verify the API endpoint is accessible

3. **PDF Processing Issues**
   - Ensure PDF files are not corrupted
   - Check file permissions
   - Some PDFs with complex layouts may not extract text properly

4. **Memory Issues**
   - Reduce `--max-chunk-size` when ingesting
   - Use smaller Ollama models
   - Close other applications to free memory

### Performance Tips

1. **For Better Speed**:
   - Use smaller models (7B vs 13B)
   - Reduce the number of search results
   - Use SSD storage for ChromaDB

2. **For Better Relevance**:
   - Use larger models
   - Increase search results
   - Fine-tune `--breakpoint-percentile` and `--max-chunk-size`

3. **For Production Use**:
   - Use persistent ChromaDB storage
   - Implement proper error handling
   - Add authentication and rate limiting

## File Structure

```
chatbot/
├── requirements.txt      # Python dependencies
├── ingest.py            # PDF ingestion script
├── chunking.py          # Fixed and semantic chunking strategies
├── test_chunking.py     # Unit tests for chunking.py
├── chatbot_app.py       # Main chatbot application
└── README.md           # This file
```

## API Endpoints

The system uses the following external APIs:

- **ChromaDB**: `http://localhost:8002` (document storage and retrieval)
- **Ollama**: `http://localhost:11434` (text generation)

## Contributing

To extend this system:

1. **Add New Document Types**: Modify `ingest.py` to support other file formats
2. **Customize RAG Prompt**: Edit the `create_rag_prompt` method in `chatbot_app.py`
3. **Add New Models**: Update the models list and ensure they're installed in Ollama
4. **Enhance UI**: Modify the Gradio interface in `chatbot_app.py`

## License

This project is open source. Feel free to modify and distribute according to your needs.
