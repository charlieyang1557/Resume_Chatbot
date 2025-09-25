"""
Retriever module for RAG Resume Q&A bot.

Handles semantic search and retrieval of relevant context chunks.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from data_loader import CorpusRecord
from index_builder import FAISSIndexBuilder


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with record and relevance score."""
    
    record: CorpusRecord
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.record.id,
            "source": self.record.source,
            "section": self.record.section,
            "date_range": self.record.date_range,
            "skills": self.record.skills,
            "text": self.record.text,
            "url": self.record.url,
            "score": self.score
        }


class DocumentRetriever:
    """Retrieves relevant documents using semantic search."""
    
    def __init__(self, index_builder: FAISSIndexBuilder):
        self.index_builder = index_builder
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0,
        source_filter: Optional[str] = None,
        skills_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum relevance score
            source_filter: Filter by source (resume/website/linkedin)
            skills_filter: Filter by skills
            
        Returns:
            List of SearchResult objects
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        # Search in FAISS index
        results = self.index_builder.search(
            query=query,
            top_k=top_k * 2,  # Get more results for filtering
            score_threshold=score_threshold
        )
        
        # Convert to SearchResult objects
        search_results = [
            SearchResult(record=record, score=score)
            for record, score in results
        ]
        
        # Apply filters
        if source_filter:
            search_results = [
                result for result in search_results
                if result.record.source == source_filter
            ]
            logger.info(f"Filtered to {len(search_results)} results from source: {source_filter}")
        
        if skills_filter:
            skill_set = set(skill.lower() for skill in skills_filter)
            filtered_results = []
            for result in search_results:
                record_skills = set(skill.lower() for skill in result.record.skills)
                if skill_set.intersection(record_skills):
                    filtered_results.append(result)
            search_results = filtered_results
            logger.info(f"Filtered to {len(search_results)} results with skills: {skills_filter}")
        
        # Return top_k results
        return search_results[:top_k]
    
    def retrieve_for_context(
        self, 
        query: str, 
        top_k: int = 3,
        min_score: float = 0.1
    ) -> List[SearchResult]:
        """
        Retrieve documents specifically for context in RAG pipeline.
        
        Args:
            query: Search query
            top_k: Number of context chunks (default 3)
            min_score: Minimum relevance score for context
            
        Returns:
            List of SearchResult objects for context
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=min_score
        )
        
        if not results:
            logger.warning(f"No relevant context found for query: {query[:50]}...")
        
        return results
    
    def get_context_text(self, results: List[SearchResult]) -> str:
        """
        Format search results as context text for LLM.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for result in results:
            # Create citation reference
            citation = f"[{result.record.source}:{result.record.section}]"
            context_parts.append(f"{citation}\n{result.record.text}")
        
        return "\n\n".join(context_parts)
    
    def get_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Extract source metadata from search results.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            List of source dictionaries
        """
        sources = []
        for result in results:
            source_info = {
                "id": result.record.id,
                "source": result.record.source,
                "section": result.record.section,
                "date_range": result.record.date_range,
                "skills": result.record.skills,
                "relevance_score": result.score
            }
            if result.record.url:
                source_info["url"] = result.record.url
            sources.append(source_info)
        
        return sources


class HybridRetriever:
    """Hybrid retriever combining semantic search with keyword matching."""
    
    def __init__(self, retriever: DocumentRetriever, keyword_boost: float = 1.1):
        self.retriever = retriever
        self.keyword_boost = keyword_boost
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[SearchResult]:
        """
        Retrieve documents with keyword boosting.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of SearchResult objects with boosted scores
        """
        # Get semantic search results
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k * 2,  # Get more for boosting
            score_threshold=score_threshold
        )
        
        # Apply keyword boosting
        query_words = set(query.lower().split())
        boosted_results = []
        
        for result in results:
            # Check for keyword matches in text
            text_words = set(result.record.text.lower().split())
            keyword_matches = len(query_words.intersection(text_words))
            
            # Boost score if keywords found
            if keyword_matches > 0:
                boost_factor = min(self.keyword_boost, 1.0 + (keyword_matches * 0.05))
                result.score *= boost_factor
            
            boosted_results.append(result)
        
        # Sort by boosted scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        
        return boosted_results[:top_k]


def main():
    """CLI test for retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test document retriever")
    parser.add_argument("index_path", help="Path to FAISS index directory")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold")
    parser.add_argument("--source", help="Filter by source")
    parser.add_argument("--skills", help="Comma-separated skills to filter by")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval")
    parser.add_argument("--context", action="store_true", help="Show formatted context")
    
    args = parser.parse_args()
    
    try:
        # Load index
        from index_builder import FAISSIndexBuilder
        builder = FAISSIndexBuilder()
        builder.load(f"{args.index_path}/index.faiss", f"{args.index_path}/metadata.pkl")
        
        # Create retriever
        retriever = DocumentRetriever(builder)
        if args.hybrid:
            retriever = HybridRetriever(retriever)
        
        # Parse filters
        skills_filter = None
        if args.skills:
            skills_filter = [s.strip() for s in args.skills.split(",")]
        
        # Retrieve documents
        results = retriever.retrieve(
            query=args.query,
            top_k=args.top_k,
            score_threshold=args.min_score,
            source_filter=args.source,
            skills_filter=skills_filter
        )
        
        print(f"🔍 Query: {args.query}")
        print(f"📊 Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f}")
            print(f"   ID: {result.record.id}")
            print(f"   Source: {result.record.source}")
            print(f"   Section: {result.record.section}")
            if result.record.date_range:
                print(f"   Date: {result.record.date_range}")
            print(f"   Skills: {', '.join(result.record.skills[:5])}")
            print(f"   Text: {result.record.text[:200]}...")
        
        # Show context if requested
        if args.context and results:
            print(f"\n📝 Context for LLM:")
            context = retriever.get_context_text(results)
            print(context)
        
        # Show sources
        sources = retriever.get_sources(results)
        print(f"\n📚 Sources:")
        for source in sources:
            print(f"   - {source['source']}:{source['section']} (score: {source['relevance_score']:.3f})")
    
    except Exception as e:
        logger.error(f"Error testing retriever: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
