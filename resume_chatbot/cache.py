"""SQLite-based caching system for query responses."""

from __future__ import annotations

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    """Cache entry for storing query responses."""
    query_hash: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    response_time: float
    timestamp: datetime
    model_used: str
    prompt_type: str


class QueryCache:
    """SQLite-based cache for storing and retrieving query responses."""
    
    def __init__(self, cache_path: Path = Path("cache/resume_chatbot.db")):
        self.cache_path = cache_path.expanduser().resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            # Create cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources TEXT NOT NULL,
                    response_time REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    prompt_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create usage statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    response_time REAL,
                    cache_hit BOOLEAN NOT NULL,
                    FOREIGN KEY (query_hash) REFERENCES query_cache (query_hash)
                )
            """)
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON query_cache (created_at)
            """)
            
            conn.commit()
    
    def _hash_query(self, query: str, prompt_type: str = "default") -> str:
        """Generate hash for query (normalized)."""
        # Normalize query for consistent hashing
        normalized = query.strip().lower()
        hash_input = f"{normalized}:{prompt_type}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get(self, query: str, prompt_type: str = "default") -> Optional[CacheEntry]:
        """Retrieve cached response for query."""
        query_hash = self._hash_query(query, prompt_type)
        
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, answer, sources, response_time, timestamp, model_used, prompt_type
                FROM query_cache 
                WHERE query_hash = ?
            """, (query_hash,))
            
            row = cursor.fetchone()
            if row:
                # Record cache hit
                cursor.execute("""
                    INSERT INTO usage_stats (query_hash, timestamp, cache_hit)
                    VALUES (?, ?, ?)
                """, (query_hash, datetime.now().isoformat(), True))
                
                conn.commit()
                
                return CacheEntry(
                    query_hash=query_hash,
                    query=row[0],
                    answer=row[1],
                    sources=json.loads(row[2]),
                    response_time=row[3],
                    timestamp=datetime.fromisoformat(row[4]),
                    model_used=row[5],
                    prompt_type=row[6]
                )
        
        # Record cache miss
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO usage_stats (query_hash, timestamp, cache_hit)
                VALUES (?, ?, ?)
            """, (query_hash, datetime.now().isoformat(), False))
            conn.commit()
        
        return None
    
    def put(
        self, 
        query: str, 
        answer: str, 
        sources: List[Dict[str, Any]], 
        response_time: float,
        model_used: str = "ollama",
        prompt_type: str = "default"
    ):
        """Store response in cache."""
        query_hash = self._hash_query(query, prompt_type)
        
        entry = CacheEntry(
            query_hash=query_hash,
            query=query,
            answer=answer,
            sources=sources,
            response_time=response_time,
            timestamp=datetime.now(),
            model_used=model_used,
            prompt_type=prompt_type
        )
        
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO query_cache 
                (query_hash, query, answer, sources, response_time, timestamp, model_used, prompt_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.query_hash,
                entry.query,
                entry.answer,
                json.dumps(entry.sources),
                entry.response_time,
                entry.timestamp.isoformat(),
                entry.model_used,
                entry.prompt_type
            ))
            
            conn.commit()
    
    def clear(self, older_than_days: Optional[int] = None):
        """Clear cache entries."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            if older_than_days:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)
                cursor.execute("""
                    DELETE FROM query_cache 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
            else:
                cursor.execute("DELETE FROM query_cache")
            
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute("SELECT COUNT(*) FROM query_cache")
            total_entries = cursor.fetchone()[0]
            
            # Cache hit rate
            cursor.execute("SELECT COUNT(*) FROM usage_stats WHERE cache_hit = 1")
            cache_hits = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM usage_stats")
            total_queries = cursor.fetchone()[0]
            
            hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0
            
            # Average response time
            cursor.execute("SELECT AVG(response_time) FROM query_cache")
            avg_response_time = cursor.fetchone()[0] or 0
            
            # Most common queries
            cursor.execute("""
                SELECT query, COUNT(*) as count 
                FROM query_cache 
                GROUP BY query_hash 
                ORDER BY count DESC 
                LIMIT 5
            """)
            common_queries = cursor.fetchall()
            
            return {
                "total_entries": total_entries,
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "hit_rate_percent": round(hit_rate, 2),
                "average_response_time": round(avg_response_time, 3),
                "common_queries": [{"query": q[0], "count": q[1]} for q in common_queries]
            }
    
    def export_for_fine_tuning(self, output_path: Path) -> int:
        """Export Q/A pairs for fine-tuning (without sensitive data)."""
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.cache_path)) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, answer, model_used, prompt_type, created_at
                FROM query_cache
                ORDER BY created_at
            """)
            
            rows = cursor.fetchall()
            
            # Export to JSONL format
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    record = {
                        "query": row[0],
                        "answer": row[1],
                        "model_used": row[2],
                        "prompt_type": row[3],
                        "timestamp": row[4],
                        "source": "resume_chatbot_cache"
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            return len(rows)


class CacheManager:
    """High-level cache management with automatic cleanup."""
    
    def __init__(self, cache_path: Path = Path("cache/resume_chatbot.db")):
        self.cache = QueryCache(cache_path)
        self.enabled = True
        self.max_age_days = 30  # Auto-cleanup after 30 days
    
    def get_or_compute(
        self,
        query: str,
        compute_func,
        prompt_type: str = "default",
        **kwargs
    ) -> tuple[str, List[Dict[str, Any]], float]:
        """Get from cache or compute and cache result."""
        if not self.enabled:
            return compute_func(query, **kwargs)
        
        # Try to get from cache
        cached = self.cache.get(query, prompt_type)
        if cached:
            print(f"🚀 Cache hit for query: {query[:50]}...")
            return cached.answer, cached.sources, cached.response_time
        
        # Compute result
        print(f"🔄 Computing result for query: {query[:50]}...")
        answer, sources, response_time = compute_func(query, **kwargs)
        
        # Cache result
        self.cache.put(
            query=query,
            answer=answer,
            sources=sources,
            response_time=response_time,
            model_used=kwargs.get('model_used', 'ollama'),
            prompt_type=prompt_type
        )
        
        return answer, sources, response_time
    
    def cleanup(self):
        """Clean up old cache entries."""
        print(f"🧹 Cleaning up cache entries older than {self.max_age_days} days...")
        self.cache.clear(older_than_days=self.max_age_days)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def export_training_data(self, output_path: Path) -> int:
        """Export training data for fine-tuning."""
        print(f"📊 Exporting training data to {output_path}")
        return self.cache.export_for_fine_tuning(output_path)


# Global cache manager instance
cache_manager = CacheManager()
