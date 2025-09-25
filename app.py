"""
FastAPI application for RAG Resume Q&A bot.

Provides REST API endpoints for asking questions about resumes.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Import our RAG components
from data_loader import load_corpus, CorpusRecord
from prompt_templates import PromptBuilder, PromptTemplates


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question about the candidate")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    template: str = Field(default="recruiter", description="Prompt template: recruiter, hiring_manager, technical, general")


class SourceResponse(BaseModel):
    id: str
    source: str
    section: str
    date_range: str
    skills: List[str]
    relevance_score: float
    url: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceResponse] = Field(..., description="Source documents used")
    response_time: float = Field(..., description="Response time in seconds")
    validation: Dict[str, Any] = Field(..., description="Response validation results")
    prompt_length: int = Field(..., description="Length of generated prompt")


class StatsResponse(BaseModel):
    total_records: int
    sources: Dict[str, int]
    top_skills: List[tuple]
    total_characters: int
    avg_chars_per_record: float


# Global state
app_state = {
    "corpus_records": [],
    "prompt_builder": None,
    "initialized": False
}


def create_app(corpus_path: str = "corpus_original.jsonl") -> FastAPI:
    """Create FastAPI application."""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="RAG Resume Q&A API",
        description="Retrieval-Augmented Generation system for answering questions about resumes",
        version="1.0.0",
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
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize the application on startup."""
        try:
            logger.info("🚀 Initializing RAG Resume Q&A API...")
            
            # Load corpus
            corpus_file = Path(corpus_path)
            if not corpus_file.exists():
                logger.error(f"Corpus file not found: {corpus_path}")
                return
            
            logger.info(f"📚 Loading corpus from: {corpus_path}")
            app_state["corpus_records"] = load_corpus(corpus_path)
            
            if not app_state["corpus_records"]:
                logger.error("No records loaded from corpus")
                return
            
            # Initialize prompt builder
            config = PromptBuilder()._get_default_config()
            config.system_prompt = PromptTemplates.recruiter_prompt()
            app_state["prompt_builder"] = PromptBuilder(config)
            
            app_state["initialized"] = True
            logger.info(f"✅ API initialized with {len(app_state['corpus_records'])} records")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize API: {e}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy" if app_state["initialized"] else "not_initialized",
            "records_loaded": len(app_state["corpus_records"]),
            "timestamp": time.time()
        }
    
    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get corpus statistics."""
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="API not initialized")
        
        records = app_state["corpus_records"]
        
        # Calculate statistics
        sources = {}
        skills_count = {}
        total_chars = 0
        
        for record in records:
            # Count by source
            sources[record.source] = sources.get(record.source, 0) + 1
            
            # Count skills
            for skill in record.skills:
                skills_count[skill] = skills_count.get(skill, 0) + 1
            
            # Count characters
            total_chars += len(record.text)
        
        top_skills = sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return StatsResponse(
            total_records=len(records),
            sources=sources,
            top_skills=top_skills,
            total_characters=total_chars,
            avg_chars_per_record=total_chars / len(records) if records else 0
        )
    
    @app.post("/ask", response_model=QuestionResponse)
    async def ask_question(request: QuestionRequest):
        """Ask a question and get an answer."""
        if not app_state["initialized"]:
            raise HTTPException(status_code=503, detail="API not initialized")
        
        start_time = time.time()
        
        try:
            # Simple keyword-based search (fallback when FAISS not available)
            search_results = _simple_search(
                request.question, 
                app_state["corpus_records"], 
                top_k=request.top_k
            )
            
            # Build prompt
            prompt_builder = app_state["prompt_builder"]
            
            # Update prompt template if requested
            if request.template != "recruiter":
                templates = PromptTemplates()
                if request.template == "hiring_manager":
                    system_prompt = templates.hiring_manager_prompt()
                elif request.template == "technical":
                    system_prompt = templates.technical_prompt()
                elif request.template == "general":
                    system_prompt = templates.general_prompt()
                else:
                    system_prompt = templates.recruiter_prompt()
                
                config = prompt_builder.config
                config.system_prompt = system_prompt
                prompt_builder = PromptBuilder(config)
            
            prompt = prompt_builder.build_prompt(request.question, search_results)
            
            # Simulate LLM response (replace with actual LLM call)
            answer = _simulate_llm_response(request.question, search_results)
            
            # Validate response
            validation = prompt_builder.validate_response(answer, has_context=bool(search_results))
            
            # Format sources
            sources = []
            for result in search_results:
                source = SourceResponse(
                    id=result.record.id,
                    source=result.record.source,
                    section=result.record.section,
                    date_range=result.record.date_range,
                    skills=result.record.skills,
                    relevance_score=result.score,
                    url=result.record.url
                )
                sources.append(source)
            
            response_time = time.time() - start_time
            
            return QuestionResponse(
                answer=answer,
                sources=sources,
                response_time=response_time,
                validation=validation,
                prompt_length=len(prompt)
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "RAG Resume Q&A API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "ask": "/ask"
        }
    
    return app


def _simple_search(query: str, records: List[CorpusRecord], top_k: int = 3):
    """Simple keyword-based search as fallback when FAISS not available."""
    from retriever import SearchResult
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_results = []
    
    for record in records:
        text_lower = record.text.lower()
        text_words = set(text_lower.split())
        
        # Calculate simple score based on word overlap
        overlap = len(query_words.intersection(text_words))
        score = overlap / len(query_words) if query_words else 0
        
        # Boost score for exact phrase matches
        if query_lower in text_lower:
            score += 0.5
        
        # Boost score for skills matches
        skills_lower = [skill.lower() for skill in record.skills]
        skill_matches = sum(1 for word in query_words if any(word in skill for skill in skills_lower))
        if skill_matches > 0:
            score += skill_matches * 0.1
        
        if score > 0:
            scored_results.append(SearchResult(record=record, score=score))
    
    # Sort by score and return top_k
    scored_results.sort(key=lambda x: x.score, reverse=True)
    return scored_results[:top_k]


def _simulate_llm_response(question: str, search_results) -> str:
    """Simulate LLM response for testing purposes."""
    if not search_results:
        return "I don't have that information in Charlie's records."
    
    # Simple response simulation based on question content
    if "ptc onshape" in question.lower() or "onshape" in question.lower():
        return ("Based on Charlie's experience at PTC Onshape, he worked on building unsupervised anomaly detection systems for API telemetry. "
               "He implemented multiple algorithms including Prophet for time series forecasting, Isolation Forest for outlier detection, and LSTM-AE for deep learning-based anomaly detection [resume:Experience > PTC Onshape]. "
               "The system achieved 95% accuracy in detecting API performance anomalies and reduced manual monitoring by 80%.")
    
    elif "pinecone" in question.lower():
        return ("At Pinecone, Charlie designed and built Book of Business and Account 360 dashboards using SQL and Sigma, improving sales operations by approximately 15% [resume:Experience > Pinecone]. "
               "He also conducted churn analysis using Python and Random Forest, identified 5 key metrics, and set up alerts that reduced churn by about 10%.")
    
    elif "skills" in question.lower() or "technologies" in question.lower():
        skills = []
        for result in search_results:
            skills.extend(result.record.skills)
        unique_skills = list(set(skills))[:10]  # Top 10 unique skills
        return (f"Charlie has experience with a wide range of technologies including: {', '.join(unique_skills)} [resume:Skills & Tools]. "
               "His expertise spans programming languages (Python, SQL, R), machine learning frameworks (Scikit-learn, TensorFlow), cloud platforms (AWS, GCP), and data visualization tools (Tableau, Power BI, Looker).")
    
    elif "education" in question.lower():
        return ("Charlie holds a Master of Science in Statistics from University of California, Davis (2021-2023) and a Bachelor of Science in Statistics and Economics from the same university (2017-2021) [resume:Education]. "
               "His relevant coursework included Advanced Statistical Computing, Algorithm Design & Analysis, Econometrics, and Statistical Machine Learning.")
    
    else:
        # Generic response based on context
        if search_results:
            result = search_results[0]
            return (f"Based on Charlie's background, {result.record.text[:200]}... "
                   f"[{result.record.source}:{result.record.section}]")
        else:
            return "I don't have that information in Charlie's records."


# Create default app instance
if FASTAPI_AVAILABLE:
    app = create_app()


def main():
    """Run the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    import uvicorn
    
    print("🚀 Starting RAG Resume Q&A API...")
    print("📚 Loading corpus and initializing components...")
    
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
