"""
Simple FastAPI application for RAG Resume Q&A bot (without heavy dependencies).

Provides REST API endpoints for asking questions about resumes using simple keyword search.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create dummy classes for when FastAPI is not available
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

# Simple data structures (no heavy dependencies)
from dataclasses import dataclass
from data_loader import load_corpus, CorpusRecord


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimpleSearchResult:
    """Simple search result for testing."""
    record: CorpusRecord
    score: float


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
    "initialized": False
}


def simple_search(query: str, records: List[CorpusRecord], top_k: int = 3) -> List[SimpleSearchResult]:
    """Simple keyword-based search."""
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
            scored_results.append(SimpleSearchResult(record=record, score=score))
    
    # Sort by score and return top_k
    scored_results.sort(key=lambda x: x.score, reverse=True)
    return scored_results[:top_k]


def build_simple_prompt(question: str, search_results: List[SimpleSearchResult], template: str = "recruiter") -> str:
    """Build a simple prompt without heavy dependencies."""
    
    # Simple system prompts
    system_prompts = {
        "recruiter": """You are Charlie's resume assistant helping recruiters evaluate his qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on skills, experience, and qualifications relevant to the role
5. Keep answers concise and professional (2-6 sentences)
6. Never provide salary information unless explicitly mentioned in context""",
        
        "hiring_manager": """You are an assistant helping hiring managers evaluate Charlie's qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on technical skills, project experience, and achievements
5. Provide specific examples when available
6. Keep answers detailed but concise (3-6 sentences)""",
        
        "technical": """You are a technical assistant helping evaluate Charlie's technical qualifications.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Focus on technical skills, tools, frameworks, and methodologies
5. Provide specific examples of technical work
6. Highlight relevant technical achievements""",
        
        "general": """You are a helpful assistant that answers questions about Charlie's resume.

Remember: Charlie is also known as Yutian Yang—treat references to Charlie, Yutian, or Yang as the same person.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context
2. If information is not in the context, say "I don't have that information in Charlie's records"
3. Always cite sources using [source:section] format
4. Keep answers helpful and professional
5. Never provide salary information unless explicitly mentioned in context"""
    }
    
    system_prompt = system_prompts.get(template, system_prompts["recruiter"])
    
    # Build context
    context_parts = []
    for result in search_results:
        citation = f"[{result.record.source}:{result.record.section}]"
        context_parts.append(f"{citation}\n{result.record.text}")
    
    context = "\n\n".join(context_parts) if context_parts else "No relevant information found."
    
    # Build prompt
    prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION: {question}

Please provide a helpful response based only on the information above."""
    
    return prompt


def simulate_llm_response(question: str, search_results: List[SimpleSearchResult]) -> str:
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


def validate_response(response: str, has_context: bool = True) -> Dict[str, Any]:
    """Validate response for safety."""
    validation = {
        "is_safe": True,
        "has_citations": False,
        "warnings": [],
        "suggestions": []
    }
    
    # Check for citations
    citations = re.findall(r'\[([^\]]+)\]', response)
    if citations:
        validation["has_citations"] = True
    
    # Check for unsafe patterns
    unsafe_patterns = [r'\$\d+', r'\b\d{3}-\d{2}-\d{4}\b']
    for pattern in unsafe_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            validation["is_safe"] = False
            validation["warnings"].append(f"Contains sensitive info: {pattern}")
    
    # Check for disclaimers
    disclaimer_indicators = ["I don't have that information", "not mentioned in", "not in the records"]
    has_disclaimer = any(indicator in response for indicator in disclaimer_indicators)
    if not has_disclaimer and not has_context:
        validation["warnings"].append("May contain hallucinated information")
    
    return validation


def create_simple_app(corpus_path: str = "corpus_original.jsonl"):
    """Create simple FastAPI application."""
    
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI not available. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="Simple RAG Resume Q&A API",
        description="Simple Retrieval-Augmented Generation system for answering questions about resumes",
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
            logger.info("🚀 Initializing Simple RAG Resume Q&A API...")
            
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
            # Search for relevant documents
            search_results = simple_search(
                request.question, 
                app_state["corpus_records"], 
                top_k=request.top_k
            )
            
            # Build prompt
            prompt = build_simple_prompt(request.question, search_results, request.template)
            
            # Simulate LLM response
            answer = simulate_llm_response(request.question, search_results)
            
            # Validate response
            validation = validate_response(answer, has_context=bool(search_results))
            
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
            "message": "Simple RAG Resume Q&A API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "ask": "/ask"
        }
    
    return app


# Create default app instance
if FASTAPI_AVAILABLE:
    app = create_simple_app()


def main():
    """Run the simple FastAPI application."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with: pip install fastapi uvicorn")
        return
    
    import uvicorn
    
    print("🚀 Starting Simple RAG Resume Q&A API...")
    print("📚 Loading corpus and initializing components...")
    
    uvicorn.run(
        "simple_app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
