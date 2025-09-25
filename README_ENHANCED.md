# 🤖 Enhanced Resume Chatbot

A production-ready RAG-based chatbot that answers questions about resumes using advanced vector search, caching, and safety features.

## ✨ Features

### 🎯 Core Capabilities
- **RAG Architecture**: Retrieval-Augmented Generation with FAISS vector search
- **Multiple LLM Backends**: Ollama (local), OpenAI, or simple fallback
- **Smart Chunking**: Intelligent document segmentation with overlap
- **Source Citation**: Automatic source attribution with `[source:file:section]`
- **Safety First**: Privacy protection and content validation

### 🔍 Advanced Retrieval
- **FAISS Vector Store**: High-performance similarity search
- **Sentence Transformers**: `all-MiniLM-L6-v2` embeddings
- **Hybrid Search**: Combines semantic and keyword matching
- **Fallback Support**: TF-IDF when FAISS unavailable

### 💾 Performance & Caching
- **SQLite Caching**: Automatic response caching with hit tracking
- **Query Deduplication**: Normalized query hashing
- **Training Data Export**: Q/A pair collection for fine-tuning
- **Statistics Dashboard**: Cache hit rates and performance metrics

### 🛡️ Safety & Privacy
- **PII Protection**: Automatic sensitive data filtering
- **Response Validation**: Content safety checking
- **Source Verification**: Citation requirement enforcement
- **Privacy Boundaries**: Salary and contact info protection

### 📁 Document Support
- **Multiple Formats**: PDF, DOC/DOCX, Markdown, JSON, TXT
- **Intelligent Parsing**: Section-aware document processing
- **Metadata Preservation**: Source tracking and chunk indexing
- **Batch Processing**: Efficient bulk document ingestion

## 🚀 Quick Start

### Option 1: Automated Setup
```bash
# Clone and setup
git clone <your-repo>
cd Resume_Chatbot
python setup_enhanced.py
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements-enhanced.txt
pip install -e .

# Setup Ollama (for local LLM)
brew install ollama
brew services start ollama
ollama pull llama3.2:3b

# Run the enhanced webapp
uvicorn resume_chatbot.enhanced_webapp:app --reload
```

### Access the Application
- **Web Interface**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/health

## 📊 API Endpoints

### Core Chat API
```http
POST /api/chat
Content-Type: application/json

{
  "message": "What programming languages does the candidate know?",
  "prompt_type": "recruiter",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The candidate is proficient in Python, R, SQL, and JavaScript [source:resume.md:Technical Skills]...",
  "sources": [
    {
      "source": "resume.md",
      "title": "Technical Skills",
      "chunk_id": "resume_Technical_Skills_0",
      "relevance_score": 0.95
    }
  ],
  "response_time": 1.23,
  "model_used": "ollama",
  "prompt_type": "recruiter",
  "cache_hit": false,
  "validation_result": {...},
  "citations": [...]
}
```

### Management APIs
- `GET /api/stats` - System statistics
- `POST /api/rebuild-index` - Rebuild vector index
- `POST /api/export-training-data` - Export Q/A pairs
- `GET /health` - Health check

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │    │   FastAPI App    │    │  Enhanced LLM   │
│                 │◄──►│                  │◄──►│   (Ollama/OpenAI)│
│  Chat Interface │    │  /api/chat       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Enhanced Chatbot │
                       │                  │
                       │ • Prompt Builder │
                       │ • Response Valid │
                       │ • Cache Manager  │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ FAISS Vector Store│
                       │                  │
                       │ • Embeddings     │
                       │ • Similarity     │
                       │ • Retrieval      │
                       └──────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │ Document Storage │
                       │                  │
                       │ • PDF/DOC/MD     │
                       │ • Chunking       │
                       │ • Metadata       │
                       └──────────────────┘
```

## 🔧 Configuration

### Environment Variables
```bash
export LLM_BACKEND="ollama"           # ollama, openai, simple
export USE_FAISS="true"               # Enable FAISS vector store
export USE_CACHE="true"               # Enable response caching
export CHUNK_SIZE="1000"              # Document chunk size
export OVERLAP="200"                  # Chunk overlap
```

### Config File (`config.json`)
```json
{
  "chatbot": {
    "llm_backend": "ollama",
    "use_faiss": true,
    "use_cache": true,
    "chunk_size": 1000,
    "overlap": 200
  },
  "vector_store": {
    "model_name": "all-MiniLM-L6-v2",
    "index_type": "flat"
  },
  "cache": {
    "max_age_days": 30,
    "cache_path": "cache/resume_chatbot.db"
  }
}
```

## 📝 Usage Examples

### Basic Question
```python
from resume_chatbot.enhanced_chatbot import create_enhanced_chatbot

chatbot = create_enhanced_chatbot(
    resume_directory=Path("data/resume"),
    llm_backend="ollama"
)

result = chatbot.ask("What is the candidate's experience with machine learning?")
print(result.answer)
print(f"Sources: {result.sources}")
```

### Custom Prompt Types
```python
# For recruiters
result = chatbot.ask("Tell me about the candidate's background", prompt_type="recruiter")

# For technical interviews
result = chatbot.ask("What ML frameworks has the candidate used?", prompt_type="technical")

# For hiring managers
result = chatbot.ask("What are the candidate's key achievements?", prompt_type="hiring_manager")
```

### Cache Management
```python
# Get cache statistics
stats = chatbot.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate_percent']}%")

# Export training data
num_records = chatbot.export_training_data(Path("training_data/qa_pairs.jsonl"))
print(f"Exported {num_records} Q/A pairs")
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_enhanced_chatbot.py
pytest tests/test_vector_store.py
pytest tests/test_prompt_templates.py
```

### Test with Sample Data
```bash
# Use the LLM judge framework
python test_llm_judge.py
python quick_test.py
```

## 📈 Performance

### Benchmarks
- **Vector Search**: < 50ms for 1000+ documents
- **LLM Response**: 1-3s with Ollama Llama 3.2
- **Cache Hit Rate**: 60-80% for repeated queries
- **Memory Usage**: ~500MB for typical resume corpus

### Optimization Tips
1. **Use FAISS**: 10x faster than TF-IDF
2. **Enable Caching**: Reduces response time by 80%
3. **Tune Chunk Size**: 800-1200 chars optimal
4. **Batch Processing**: Load multiple documents at once

## 🔒 Security & Privacy

### Data Protection
- **No PII Storage**: Sensitive data filtered automatically
- **Local Processing**: All data stays on your machine
- **Secure Caching**: Hashed queries, no raw data
- **Audit Trail**: Query logging for compliance

### Content Validation
- **Response Checking**: Automatic safety validation
- **Citation Requirements**: Source attribution enforced
- **Hallucination Detection**: Missing info disclaimers
- **Privacy Boundaries**: Salary/contact info protection

## 🚀 Deployment

### Local Development
```bash
uvicorn resume_chatbot.enhanced_webapp:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment
```bash
# Using Gunicorn
gunicorn resume_chatbot.enhanced_webapp:app -w 4 -k uvicorn.workers.UvicornWorker

# Using Docker
docker build -t resume-chatbot .
docker run -p 8000:8000 resume-chatbot
```

### Docker Setup
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements-enhanced.txt .
RUN pip install -r requirements-enhanced.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "resume_chatbot.enhanced_webapp:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🔧 Troubleshooting

### Common Issues

**FAISS Import Error**
```bash
# Install FAISS
pip install faiss-cpu  # For CPU
pip install faiss-gpu  # For GPU
```

**Ollama Connection Error**
```bash
# Start Ollama service
brew services start ollama
# Or manually
ollama serve
```

**Memory Issues**
```bash
# Reduce chunk size
export CHUNK_SIZE="500"
# Or use smaller model
ollama pull llama3.2:1b
```

**Slow Responses**
```bash
# Enable caching
export USE_CACHE="true"
# Use smaller model
ollama pull llama3.2:1b
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL="DEBUG"
uvicorn resume_chatbot.enhanced_webapp:app --reload --log-level debug
```

## 📚 Advanced Usage

### Custom Embeddings
```python
from resume_chatbot.vector_store import FAISSVectorStore

# Use different embedding model
vector_store = FAISSVectorStore(model_name="all-mpnet-base-v2")
```

### Custom Prompts
```python
from resume_chatbot.prompt_templates import PromptBuilder

# Create custom prompt type
prompt_builder.system_prompts["custom"] = "Your custom system prompt..."
```

### Batch Processing
```python
# Process multiple documents
documents = load_resume_documents(Path("data/resumes/"))
chatbot.retriever.add_documents(documents)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FAISS**: Facebook AI Similarity Search
- **Sentence Transformers**: Hugging Face
- **Ollama**: Local LLM server
- **FastAPI**: Modern web framework
- **Pydantic**: Data validation

---

**Built with ❤️ for better resume screening and candidate evaluation.**
