"""
FAISS index builder for RAG Resume Q&A bot.

Builds and manages FAISS vector index from corpus data using sentence transformers.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    SentenceTransformer = None

from data_loader import CorpusRecord, load_corpus


logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Builds and manages FAISS vector index for semantic search."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "flat"
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS and sentence-transformers required. Install with:\n"
                "pip install faiss-cpu sentence-transformers"
            )
        
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Metadata storage
        self.records: List[CorpusRecord] = []
        self.record_metadata: List[Dict[str, Any]] = []
        self._is_trained = False
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def build_from_corpus(self, corpus_path: str | Path) -> None:
        """Build index from corpus.jsonl file."""
        logger.info(f"Building index from corpus: {corpus_path}")
        
        # Load corpus
        records = load_corpus(corpus_path)
        if not records:
            raise ValueError(f"No records found in corpus: {corpus_path}")
        
        self.build_from_records(records)
    
    def build_from_records(self, records: List[CorpusRecord]) -> None:
        """Build index from list of CorpusRecord objects."""
        if not records:
            raise ValueError("No records provided")
        
        logger.info(f"Building index from {len(records)} records")
        
        # Extract text for embedding
        texts = [record.text for record in records]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Normalize for cosine similarity
        embeddings = self._normalize_embeddings(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self._is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)
            self._is_trained = True
        
        # Add to index
        self.index.add(embeddings)
        
        # Store records and metadata
        self.records = records
        self.record_metadata = [record.to_dict() for record in records]
        
        logger.info(f"✅ Index built successfully with {len(records)} records")
    
    def save(self, index_path: str | Path, metadata_path: str | Path = None) -> None:
        """Save FAISS index and metadata to disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path) if metadata_path else index_path.with_suffix('.metadata.pkl')
        
        # Create output directory
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        logger.info(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self._is_trained,
            'records': self.record_metadata,
            'num_records': len(self.records)
        }
        
        logger.info(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("✅ Index and metadata saved successfully")
    
    def load(self, index_path: str | Path, metadata_path: str | Path = None) -> None:
        """Load FAISS index and metadata from disk."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path) if metadata_path else index_path.with_suffix('.metadata.pkl')
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        logger.info(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self._is_trained = metadata['is_trained']
        self.record_metadata = metadata['records']
        
        # Reconstruct records
        self.records = [CorpusRecord.from_dict(record_data) for record_data in self.record_metadata]
        
        # Reload embedding model
        self.embedding_model = SentenceTransformer(self.model_name)
        
        logger.info(f"✅ Index loaded successfully with {len(self.records)} records")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[CorpusRecord, float]]:
        """Search for similar records."""
        if not self.records:
            logger.warning("No records in index")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.records)))
        
        # Filter by score threshold and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            if score >= score_threshold:
                results.append((self.records[idx], float(score)))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_records": len(self.records),
            "model_name": self.model_name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self._is_trained,
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.records)
        }


def build_index_from_corpus(
    corpus_path: str | Path,
    output_dir: str | Path = "data/faiss_index",
    model_name: str = "all-MiniLM-L6-v2",
    index_type: str = "flat"
) -> None:
    """
    Convenience function to build and save FAISS index from corpus.
    
    Args:
        corpus_path: Path to corpus.jsonl file
        output_dir: Directory to save index files
        model_name: Sentence transformer model name
        index_type: FAISS index type ("flat" or "ivf")
    """
    output_dir = Path(output_dir)
    
    # Build index
    builder = FAISSIndexBuilder(model_name=model_name, index_type=index_type)
    builder.build_from_corpus(corpus_path)
    
    # Save index
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.pkl"
    builder.save(index_path, metadata_path)
    
    # Print stats
    stats = builder.get_stats()
    print(f"✅ Index built successfully!")
    print(f"   Records: {stats['num_records']}")
    print(f"   Model: {stats['model_name']}")
    print(f"   Type: {stats['index_type']}")
    print(f"   Saved to: {output_dir}")


def main():
    """CLI for building FAISS index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus")
    parser.add_argument("corpus_path", help="Path to corpus.jsonl file")
    parser.add_argument("--output", "-o", default="data/faiss_index", help="Output directory")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--index-type", default="flat", choices=["flat", "ivf"], help="FAISS index type")
    parser.add_argument("--test", action="store_true", help="Test search after building")
    
    args = parser.parse_args()
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Install with: pip install faiss-cpu sentence-transformers")
        return
    
    try:
        # Build index
        build_index_from_corpus(
            corpus_path=args.corpus_path,
            output_dir=args.output,
            model_name=args.model,
            index_type=args.index_type
        )
        
        # Test search if requested
        if args.test:
            print("\n🔍 Testing search...")
            builder = FAISSIndexBuilder(model_name=args.model, index_type=args.index_type)
            builder.load(args.output + "/index.faiss", args.output + "/metadata.pkl")
            
            # Test query
            test_query = "What did Charlie work on at PTC Onshape?"
            results = builder.search(test_query, top_k=3)
            
            print(f"Query: {test_query}")
            print(f"Found {len(results)} results:")
            
            for i, (record, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   ID: {record.id}")
                print(f"   Source: {record.source}")
                print(f"   Section: {record.section}")
                print(f"   Text: {record.text[:150]}...")
    
    except Exception as e:
        logger.error(f"Error building index: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
