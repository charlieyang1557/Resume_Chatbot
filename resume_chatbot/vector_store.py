"""FAISS-based vector store for semantic search."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from .enhanced_data_loader import EnhancedDocument


class FAISSVectorStore:
    """FAISS-based vector store for semantic document search."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "flat"
    ):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS and sentence-transformers required. Install with:\n"
                "pip install faiss-cpu sentence-transformers torch"
            )
        
        self.model_name = model_name
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Document storage
        self.documents: List[EnhancedDocument] = []
        self.document_metadata: List[Dict[str, Any]] = []
        
        self._is_trained = False
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def add_documents(self, documents: List[EnhancedDocument]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        print(f"📚 Adding {len(documents)} documents to vector store...")
        
        # Extract content for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        print("🔄 Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normalize for cosine similarity
        embeddings = self._normalize_embeddings(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self._is_trained:
            print("🔄 Training FAISS index...")
            self.index.train(embeddings)
            self._is_trained = True
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.document_metadata.extend([doc.metadata for doc in documents])
        
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Tuple[EnhancedDocument, float]]:
        """Search for similar documents."""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        # Filter by score threshold and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            if score >= score_threshold:
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: Path):
        """Save the vector store to disk."""
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = path.with_suffix('.faiss')
        faiss.write_index(self.index, str(faiss_path))
        
        # Save documents and metadata
        data_path = path.with_suffix('.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'is_trained': self._is_trained
            }, f)
        
        print(f"💾 Vector store saved to {path}")
    
    def load(self, path: Path):
        """Load the vector store from disk."""
        path = path.expanduser().resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")
        
        # Load FAISS index
        faiss_path = path.with_suffix('.faiss')
        self.index = faiss.read_index(str(faiss_path))
        
        # Load documents and metadata
        data_path = path.with_suffix('.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.document_metadata = data['document_metadata']
        self.model_name = data['model_name']
        self.dimension = data['dimension']
        self.index_type = data['index_type']
        self._is_trained = data['is_trained']
        
        # Reload embedding model
        self.embedding_model = SentenceTransformer(self.model_name)
        
        print(f"📂 Vector store loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "num_documents": len(self.documents),
            "index_type": self.index_type,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_trained": self._is_trained,
            "index_size": self.index.ntotal if hasattr(self.index, 'ntotal') else len(self.documents)
        }
    
    def rebuild_index(self, index_type: str = "flat"):
        """Rebuild the index with different type."""
        if not self.documents:
            print("No documents to rebuild index")
            return
        
        print(f"🔄 Rebuilding index with type: {index_type}")
        
        # Extract embeddings
        texts = [doc.content for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        embeddings = self._normalize_embeddings(embeddings)
        
        # Create new index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            self.index.train(embeddings)
            self._is_trained = True
        
        # Add embeddings
        self.index.add(embeddings)
        self.index_type = index_type
        
        print(f"✅ Index rebuilt with type: {index_type}")


class HybridRetriever:
    """Hybrid retriever combining FAISS semantic search with keyword matching."""
    
    def __init__(self, vector_store: FAISSVectorStore, keyword_boost: float = 1.2):
        self.vector_store = vector_store
        self.keyword_boost = keyword_boost
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[EnhancedDocument, float]]:
        """Hybrid search combining semantic and keyword matching."""
        # Semantic search
        semantic_results = self.vector_store.search(query, top_k=top_k * 2)
        
        # Keyword boost
        query_lower = query.lower()
        boosted_results = []
        
        for doc, score in semantic_results:
            # Boost score if query keywords appear in document
            content_lower = doc.content.lower()
            keyword_matches = sum(1 for word in query_lower.split() if word in content_lower)
            
            if keyword_matches > 0:
                score *= self.keyword_boost
            
            boosted_results.append((doc, score))
        
        # Sort by boosted scores and return top_k
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]


# Fallback retriever for when FAISS is not available
class FallbackRetriever:
    """Fallback retriever using the original TF-IDF approach."""
    
    def __init__(self):
        from .retriever import ResumeRetriever
        self.retriever = None
    
    def add_documents(self, documents: List[EnhancedDocument]):
        """Add documents using TF-IDF approach."""
        # Convert to legacy format
        from .data_loader import Document
        legacy_docs = []
        for doc in documents:
            legacy_doc = Document(content=doc.content, metadata=doc.metadata)
            legacy_docs.append(legacy_doc)
        
        self.retriever = ResumeRetriever(legacy_docs)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[EnhancedDocument, float]]:
        """Search using TF-IDF."""
        if not self.retriever:
            return []
        
        results = self.retriever.retrieve(query, top_k=top_k)
        
        # Convert back to enhanced format
        enhanced_results = []
        for result in results:
            enhanced_doc = EnhancedDocument(
                content=result.document.content,
                metadata=result.document.metadata,
                chunk_id=f"tfidf_{hash(result.document.content) % 10000}",
                source_hash="tfidf"
            )
            enhanced_results.append((enhanced_doc, result.score))
        
        return enhanced_results


def create_retriever(use_faiss: bool = True, **kwargs) -> FAISSVectorStore | FallbackRetriever:
    """Create appropriate retriever based on available dependencies."""
    if use_faiss and FAISS_AVAILABLE:
        return FAISSVectorStore(**kwargs)
    else:
        print("⚠️  FAISS not available, falling back to TF-IDF")
        return FallbackRetriever()
