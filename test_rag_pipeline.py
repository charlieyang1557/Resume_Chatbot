#!/usr/bin/env python3
"""
CLI test script for RAG Resume Q&A pipeline.

Tests the complete pipeline: data loading -> indexing -> retrieval -> prompt building.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Import our modules
from data_loader import load_corpus, CorpusLoader
from index_builder import FAISSIndexBuilder
from retriever import DocumentRetriever, HybridRetriever
from prompt_templates import PromptBuilder, PromptTemplates


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for resume Q&A."""
    
    def __init__(self, corpus_path: str, index_path: str = "data/faiss_index"):
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)
        
        # Initialize components
        self.index_builder: FAISSIndexBuilder = None
        self.retriever: DocumentRetriever = None
        self.prompt_builder: PromptBuilder = None
        
        # Load pipeline
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load or build the complete pipeline."""
        logger.info("🚀 Initializing RAG pipeline...")
        
        # Check if index exists
        index_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            logger.info("📂 Loading existing FAISS index...")
            self.index_builder = FAISSIndexBuilder()
            self.index_builder.load(index_file, metadata_file)
        else:
            logger.info("🔨 Building new FAISS index...")
            self.index_builder = FAISSIndexBuilder()
            self.index_builder.build_from_corpus(self.corpus_path)
            
            # Save index
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.index_builder.save(index_file, metadata_file)
        
        # Initialize retriever
        self.retriever = DocumentRetriever(self.index_builder)
        
        # Initialize prompt builder with recruiter template
        config = PromptBuilder()._get_default_config()
        config.system_prompt = PromptTemplates.recruiter_prompt()
        self.prompt_builder = PromptBuilder(config)
        
        logger.info("✅ RAG pipeline initialized successfully!")
    
    def ask(self, question: str, top_k: int = 3, use_hybrid: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get a complete response.
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            use_hybrid: Whether to use hybrid retrieval
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        logger.info(f"❓ Question: {question[:50]}...")
        
        # Retrieve relevant documents
        if use_hybrid:
            retriever = HybridRetriever(self.retriever)
        else:
            retriever = self.retriever
        
        search_results = retriever.retrieve_for_context(question, top_k=top_k)
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(question, search_results)
        
        # Get sources
        sources = self.retriever.get_sources(search_results)
        
        # For CLI testing, we'll simulate LLM response
        # In production, this would call the actual LLM
        answer = self._simulate_llm_response(question, search_results)
        
        # Validate response
        validation = self.prompt_builder.validate_response(answer, has_context=bool(search_results))
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_results": len(search_results),
            "validation": validation,
            "prompt_length": len(prompt)
        }
    
    def _simulate_llm_response(self, question: str, search_results: List) -> str:
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
            context_text = self.retriever.get_context_text(search_results)
            return (f"Based on Charlie's background, {context_text[:200]}... "
                   f"[{search_results[0].record.source}:{search_results[0].record.section}]")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        index_stats = self.index_builder.get_stats()
        
        return {
            "corpus_path": str(self.corpus_path),
            "index_path": str(self.index_path),
            "total_records": index_stats["num_records"],
            "model_name": index_stats["model_name"],
            "index_type": index_stats["index_type"]
        }


def main():
    """Main CLI function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Resume Q&A pipeline")
    parser.add_argument("corpus_path", help="Path to corpus.jsonl file")
    parser.add_argument("--index-path", default="data/faiss_index", help="FAISS index directory")
    parser.add_argument("--question", default="What did Charlie work on at PTC Onshape?", help="Test question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of context chunks")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--stats", action="store_true", help="Show pipeline statistics")
    
    args = parser.parse_args()
    
    # Check if corpus exists
    if not Path(args.corpus_path).exists():
        print(f"❌ Corpus file not found: {args.corpus_path}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = RAGPipeline(args.corpus_path, args.index_path)
        
        # Show stats if requested
        if args.stats:
            stats = pipeline.get_stats()
            print("\n📊 Pipeline Statistics:")
            print(f"  Corpus: {stats['corpus_path']}")
            print(f"  Index: {stats['index_path']}")
            print(f"  Records: {stats['total_records']}")
            print(f"  Model: {stats['model_name']}")
            print(f"  Index Type: {stats['index_type']}")
            print()
        
        # Interactive mode
        if args.interactive:
            print("🤖 RAG Resume Q&A Bot - Interactive Mode")
            print("Type 'quit' to exit, 'stats' for statistics")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\n❓ Question: ").strip()
                    
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if question.lower() == 'stats':
                        stats = pipeline.get_stats()
                        print(f"📊 Total records: {stats['total_records']}")
                        continue
                    
                    if not question:
                        continue
                    
                    # Get answer
                    result = pipeline.ask(question, top_k=args.top_k, use_hybrid=args.hybrid)
                    
                    # Display result
                    print(f"\n🤖 Answer: {result['answer']}")
                    print(f"\n📚 Sources ({result['search_results']} found):")
                    for source in result['sources']:
                        print(f"   - {source['source']}:{source['section']} (score: {source['relevance_score']:.3f})")
                    
                    if result['validation']['warnings']:
                        print(f"\n⚠️  Warnings: {', '.join(result['validation']['warnings'])}")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        else:
            # Single question mode
            print(f"🔍 Testing question: {args.question}")
            result = pipeline.ask(args.question, top_k=args.top_k, use_hybrid=args.hybrid)
            
            print(f"\n🤖 Answer: {result['answer']}")
            print(f"\n📚 Sources ({result['search_results']} found):")
            for source in result['sources']:
                print(f"   - {source['source']}:{source['section']} (score: {source['relevance_score']:.3f})")
                if source.get('skills'):
                    print(f"     Skills: {', '.join(source['skills'][:5])}")
            
            print(f"\n📊 Response Stats:")
            print(f"   Prompt length: {result['prompt_length']} chars")
            print(f"   Validation: {'✅ Safe' if result['validation']['is_safe'] else '⚠️ Issues'}")
            if result['validation']['warnings']:
                print(f"   Warnings: {', '.join(result['validation']['warnings'])}")
        
        print("\n✅ Pipeline test completed successfully!")
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
