"""Enhanced resume chatbot with FAISS, caching, and improved prompting."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from .enhanced_data_loader import load_resume_documents, EnhancedDocument
from .vector_store import create_retriever, FAISSVectorStore, FallbackRetriever
from .prompt_templates import prompt_builder, response_validator, citation_extractor
from .cache import cache_manager
from .llm import create_llm, BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnhancedChatResult:
    """Enhanced chat result with comprehensive metadata."""
    answer: str
    sources: List[Dict[str, Any]]
    response_time: float
    model_used: str
    prompt_type: str
    cache_hit: bool
    validation_result: Dict[str, Any]
    citations: List[Dict[str, str]]


class EnhancedResumeChatbot:
    """Enhanced resume chatbot with FAISS, caching, and safety features."""
    
    def __init__(
        self,
        resume_directory: Path,
        vector_store_path: Optional[Path] = None,
        llm_backend: str = "ollama",
        use_faiss: bool = True,
        use_cache: bool = True,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        self.resume_directory = resume_directory
        self.vector_store_path = vector_store_path or Path("data/vector_store")
        self.llm_backend = llm_backend
        self.use_faiss = use_faiss
        self.use_cache = use_cache
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize components
        self.retriever = None
        self.llm = None
        self.documents = []
        
        # Load and index documents
        self._initialize()
    
    def _initialize(self):
        """Initialize the chatbot by loading documents and building index."""
        logger.info("🚀 Initializing Enhanced Resume Chatbot...")
        
        # Load documents
        logger.info(f"📚 Loading documents from {self.resume_directory}")
        self.documents = load_resume_documents(
            self.resume_directory, 
            chunk_size=self.chunk_size, 
            overlap=self.overlap
        )
        
        if not self.documents:
            logger.warning("No documents loaded!")
            return
        
        logger.info(f"✅ Loaded {len(self.documents)} document chunks")
        
        # Initialize retriever
        logger.info(f"🔍 Initializing retriever (FAISS: {self.use_faiss})")
        self.retriever = create_retriever(use_faiss=self.use_faiss)
        
        # Try to load existing vector store
        if self.use_faiss and isinstance(self.retriever, FAISSVectorStore):
            if self.vector_store_path.exists():
                try:
                    self.retriever.load(self.vector_store_path)
                    logger.info("📂 Loaded existing vector store")
                    return
                except Exception as e:
                    logger.warning(f"Could not load vector store: {e}")
        
        # Build index from documents
        logger.info("🔨 Building vector index...")
        if hasattr(self.retriever, 'add_documents'):
            self.retriever.add_documents(self.documents)
        else:
            # Fallback retriever
            self.retriever.add_documents(self.documents)
        
        # Save vector store
        if self.use_faiss and isinstance(self.retriever, FAISSVectorStore):
            self.retriever.save(self.vector_store_path)
            logger.info(f"💾 Saved vector store to {self.vector_store_path}")
        
        # Initialize LLM
        logger.info(f"🤖 Initializing LLM backend: {self.llm_backend}")
        self.llm = create_llm(self.llm_backend)
        
        logger.info("✅ Enhanced Resume Chatbot initialized successfully!")
    
    def _compute_response(self, question: str, prompt_type: str = "recruiter") -> Tuple[str, List[Dict[str, Any]], float]:
        """Compute response without caching."""
        start_time = time.time()
        
        # Retrieve relevant documents
        if hasattr(self.retriever, 'search'):
            retrieved_docs = self.retriever.search(question, top_k=5)
            documents = [doc for doc, score in retrieved_docs]
        else:
            # Fallback retriever
            results = self.retriever.search(question, top_k=5)
            documents = [doc for doc, score in results]
        
        # Build prompt
        prompt = prompt_builder.build_prompt(
            query=question,
            documents=documents,
            prompt_type=prompt_type
        )
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Extract citations
        citations = citation_extractor.extract_citations(response)
        
        # Format sources for API
        sources = []
        for doc in documents:
            source_info = {
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", "content"),
                "chunk_id": doc.chunk_id,
                "relevance_score": 0.0  # Could be enhanced with actual scores
            }
            sources.append(source_info)
        
        # Add citation sources
        for citation in citations:
            if not any(s["source"] == citation["source"] for s in sources):
                sources.append({
                    "source": citation["source"],
                    "title": citation["section"],
                    "chunk_id": "citation",
                    "relevance_score": 0.0
                })
        
        response_time = time.time() - start_time
        
        return response, sources, response_time
    
    def ask(
        self, 
        question: str, 
        prompt_type: str = "recruiter",
        top_k: int = 5,
        **kwargs
    ) -> EnhancedChatResult:
        """Ask a question and get enhanced response."""
        logger.info(f"❓ Question: {question[:100]}...")
        
        # Use cache if enabled
        if self.use_cache:
            answer, sources, response_time = cache_manager.get_or_compute(
                query=question,
                compute_func=self._compute_response,
                prompt_type=prompt_type,
                model_used=self.llm_backend,
                **kwargs
            )
            cache_hit = cache_manager.cache.get(question, prompt_type) is not None
        else:
            answer, sources, response_time = self._compute_response(question, prompt_type)
            cache_hit = False
        
        # Validate response
        validation_result = response_validator.validate_response(answer)
        
        # Extract citations
        citations = citation_extractor.extract_citations(answer)
        
        # Create result
        result = EnhancedChatResult(
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used=self.llm_backend,
            prompt_type=prompt_type,
            cache_hit=cache_hit,
            validation_result=validation_result,
            citations=citations
        )
        
        # Log warnings if any
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"⚠️  {warning}")
        
        logger.info(f"✅ Response generated in {response_time:.2f}s (cache: {cache_hit})")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics."""
        stats = {
            "num_documents": len(self.documents),
            "llm_backend": self.llm_backend,
            "use_faiss": self.use_faiss,
            "use_cache": self.use_cache,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap
        }
        
        # Add retriever stats
        if hasattr(self.retriever, 'get_stats'):
            stats["retriever"] = self.retriever.get_stats()
        
        # Add cache stats
        if self.use_cache:
            stats["cache"] = cache_manager.get_stats()
        
        return stats
    
    def rebuild_index(self):
        """Rebuild the vector index."""
        logger.info("🔄 Rebuilding vector index...")
        
        if self.use_faiss and isinstance(self.retriever, FAISSVectorStore):
            self.retriever.rebuild_index()
            self.retriever.save(self.vector_store_path)
            logger.info("✅ Vector index rebuilt successfully")
        else:
            logger.warning("Index rebuilding only supported for FAISS vector store")
    
    def export_training_data(self, output_path: Path) -> int:
        """Export Q/A pairs for fine-tuning."""
        if not self.use_cache:
            logger.warning("Cache not enabled, cannot export training data")
            return 0
        
        return cache_manager.export_training_data(output_path)


# Factory function for easy creation
def create_enhanced_chatbot(
    resume_directory: Path,
    llm_backend: str = "ollama",
    use_faiss: bool = True,
    use_cache: bool = True,
    **kwargs
) -> EnhancedResumeChatbot:
    """Create an enhanced resume chatbot with specified configuration."""
    return EnhancedResumeChatbot(
        resume_directory=resume_directory,
        llm_backend=llm_backend,
        use_faiss=use_faiss,
        use_cache=use_cache,
        **kwargs
    )
