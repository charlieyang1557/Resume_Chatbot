"""Enhanced web application with improved API endpoints and features."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .enhanced_chatbot import EnhancedResumeChatbot, EnhancedChatResult


# Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User question")
    prompt_type: str = Field(default="recruiter", description="Prompt type: recruiter, hiring_manager, technical, general")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of documents to retrieve")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Chatbot response")
    sources: list[Dict[str, Any]] = Field(..., description="Source documents")
    response_time: float = Field(..., description="Response time in seconds")
    model_used: str = Field(..., description="LLM model used")
    prompt_type: str = Field(..., description="Prompt type used")
    cache_hit: bool = Field(..., description="Whether response was from cache")
    validation_result: Dict[str, Any] = Field(..., description="Response validation results")
    citations: list[Dict[str, str]] = Field(..., description="Extracted citations")


class StatsResponse(BaseModel):
    chatbot_stats: Dict[str, Any] = Field(..., description="Chatbot statistics")
    system_info: Dict[str, Any] = Field(..., description="System information")


# Global chatbot instance
app_state = {"chatbot": None}


def create_enhanced_app(
    *,
    resume_directory: Path = Path("data/resume"),
    llm_backend: str = "ollama",
    use_faiss: bool = True,
    use_cache: bool = True,
    chunk_size: int = 1000,
    overlap: int = 200
) -> FastAPI:
    """Create enhanced FastAPI application."""
    
    app = FastAPI(
        title="Enhanced Resume Chatbot API",
        description="RAG-based chatbot for answering questions about resumes",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize chatbot
    @app.on_event("startup")
    async def startup_event():
        """Initialize chatbot on startup."""
        try:
            print("🚀 Initializing Enhanced Resume Chatbot...")
            app_state["chatbot"] = EnhancedResumeChatbot(
                resume_directory=resume_directory,
                llm_backend=llm_backend,
                use_faiss=use_faiss,
                use_cache=use_cache,
                chunk_size=chunk_size,
                overlap=overlap
            )
            print("✅ Enhanced Resume Chatbot initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            raise
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the chat interface."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Resume Chatbot</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .container {
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    width: 90%;
                    max-width: 800px;
                    height: 80vh;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }
                
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }
                
                .header h1 {
                    font-size: 24px;
                    margin-bottom: 8px;
                }
                
                .header p {
                    opacity: 0.9;
                    font-size: 14px;
                }
                
                .controls {
                    padding: 15px 20px;
                    background: #f8f9fa;
                    border-bottom: 1px solid #e9ecef;
                    display: flex;
                    gap: 10px;
                    align-items: center;
                }
                
                .prompt-type {
                    padding: 8px 12px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    background: white;
                    font-size: 14px;
                }
                
                .chat-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    gap: 15px;
                }
                
                .message {
                    max-width: 80%;
                    padding: 12px 16px;
                    border-radius: 18px;
                    word-wrap: break-word;
                }
                
                .user-message {
                    background: #007bff;
                    color: white;
                    align-self: flex-end;
                    margin-left: auto;
                }
                
                .bot-message {
                    background: #f1f3f4;
                    color: #333;
                    align-self: flex-start;
                }
                
                .sources {
                    margin-top: 10px;
                    padding: 10px;
                    background: #e3f2fd;
                    border-radius: 8px;
                    font-size: 12px;
                }
                
                .source-item {
                    margin: 5px 0;
                    padding: 5px 8px;
                    background: white;
                    border-radius: 4px;
                    border-left: 3px solid #2196f3;
                }
                
                .input-container {
                    padding: 20px;
                    background: white;
                    border-top: 1px solid #e9ecef;
                    display: flex;
                    gap: 10px;
                }
                
                .message-input {
                    flex: 1;
                    padding: 12px 16px;
                    border: 1px solid #ddd;
                    border-radius: 25px;
                    font-size: 14px;
                    outline: none;
                }
                
                .message-input:focus {
                    border-color: #007bff;
                }
                
                .send-button {
                    padding: 12px 24px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 25px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: 500;
                }
                
                .send-button:hover {
                    background: #0056b3;
                }
                
                .send-button:disabled {
                    background: #6c757d;
                    cursor: not-allowed;
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    padding: 20px;
                    color: #666;
                }
                
                .stats {
                    font-size: 11px;
                    color: #666;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Enhanced Resume Chatbot</h1>
                    <p>Ask questions about the candidate's resume</p>
                </div>
                
                <div class="controls">
                    <label for="prompt-type">Context:</label>
                    <select id="prompt-type" class="prompt-type">
                        <option value="recruiter">Recruiter</option>
                        <option value="hiring_manager">Hiring Manager</option>
                        <option value="technical">Technical</option>
                        <option value="general">General</option>
                    </select>
                </div>
                
                <div class="chat-container" id="chat-container">
                    <div class="message bot-message">
                        👋 Hello! I'm your enhanced resume chatbot. I can answer questions about the candidate's background, skills, and experience. What would you like to know?
                    </div>
                </div>
                
                <div class="loading" id="loading">
                    <div>🔄 Thinking...</div>
                </div>
                
                <div class="input-container">
                    <input type="text" id="message-input" class="message-input" placeholder="Ask a question about the candidate..." />
                    <button id="send-button" class="send-button">Send</button>
                </div>
            </div>
            
            <script>
                const chatContainer = document.getElementById('chat-container');
                const messageInput = document.getElementById('message-input');
                const sendButton = document.getElementById('send-button');
                const loading = document.getElementById('loading');
                const promptType = document.getElementById('prompt-type');
                
                function addMessage(content, isUser, sources = null, stats = null) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                    
                    let html = content;
                    
                    if (sources && sources.length > 0) {
                        html += '<div class="sources"><strong>Sources:</strong>';
                        sources.forEach(source => {
                            html += `<div class="source-item">📄 ${source.source}: ${source.title}</div>`;
                        });
                        html += '</div>';
                    }
                    
                    if (stats) {
                        html += `<div class="stats">⏱️ ${stats.response_time.toFixed(2)}s | 🤖 ${stats.model_used} | ${stats.cache_hit ? '💾 Cached' : '🔄 Fresh'}</div>`;
                    }
                    
                    messageDiv.innerHTML = html;
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                async function sendMessage() {
                    const message = messageInput.value.trim();
                    if (!message) return;
                    
                    // Add user message
                    addMessage(message, true);
                    messageInput.value = '';
                    
                    // Show loading
                    loading.style.display = 'block';
                    sendButton.disabled = true;
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                message: message,
                                prompt_type: promptType.value
                            })
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        // Add bot response
                        addMessage(data.answer, false, data.sources, {
                            response_time: data.response_time,
                            model_used: data.model_used,
                            cache_hit: data.cache_hit
                        });
                        
                    } catch (error) {
                        addMessage(`❌ Error: ${error.message}`, false);
                    } finally {
                        loading.style.display = 'none';
                        sendButton.disabled = false;
                        messageInput.focus();
                    }
                }
                
                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                // Focus input on load
                messageInput.focus();
            </script>
        </body>
        </html>
        """
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Enhanced chat endpoint with comprehensive response."""
        if not app_state["chatbot"]:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        try:
            result: EnhancedChatResult = app_state["chatbot"].ask(
                question=request.message,
                prompt_type=request.prompt_type,
                top_k=request.top_k
            )
            
            return ChatResponse(
                answer=result.answer,
                sources=result.sources,
                response_time=result.response_time,
                model_used=result.model_used,
                prompt_type=result.prompt_type,
                cache_hit=result.cache_hit,
                validation_result=result.validation_result,
                citations=result.citations
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats():
        """Get chatbot and system statistics."""
        if not app_state["chatbot"]:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        chatbot_stats = app_state["chatbot"].get_stats()
        
        system_info = {
            "status": "running",
            "timestamp": time.time(),
            "version": "2.0.0"
        }
        
        return StatsResponse(
            chatbot_stats=chatbot_stats,
            system_info=system_info
        )
    
    @app.post("/api/rebuild-index")
    async def rebuild_index():
        """Rebuild the vector index."""
        if not app_state["chatbot"]:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        try:
            app_state["chatbot"].rebuild_index()
            return {"message": "Index rebuilt successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")
    
    @app.post("/api/export-training-data")
    async def export_training_data():
        """Export training data for fine-tuning."""
        if not app_state["chatbot"]:
            raise HTTPException(status_code=503, detail="Chatbot not initialized")
        
        try:
            output_path = Path("training_data/exported_qa_pairs.jsonl")
            num_records = app_state["chatbot"].export_training_data(output_path)
            return {
                "message": f"Exported {num_records} Q/A pairs",
                "output_path": str(output_path),
                "num_records": num_records
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "chatbot_initialized": app_state["chatbot"] is not None,
            "timestamp": time.time()
        }
    
    return app


# Create default app instance
app = create_enhanced_app()
