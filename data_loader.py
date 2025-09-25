"""
Data loader for RAG Resume Q&A bot.

Loads and preprocesses corpus.jsonl into structured records for indexing and retrieval.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class CorpusRecord:
    """Structured record for corpus data."""
    
    id: str
    source: str
    section: str
    date_range: str
    skills: List[str]
    text: str
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "source": self.source,
            "section": self.section,
            "date_range": self.date_range,
            "skills": self.skills,
            "text": self.text
        }
        if self.url:
            result["url"] = self.url
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusRecord":
        """Create CorpusRecord from dictionary."""
        return cls(
            id=data["id"],
            source=data["source"],
            section=data["section"],
            date_range=data.get("date_range", ""),
            skills=data.get("skills", []),
            text=data["text"],
            url=data.get("url")
        )


class CorpusLoader:
    """Loads and validates corpus data from JSONL files."""
    
    def __init__(self, corpus_path: Path):
        self.corpus_path = Path(corpus_path)
        self.records: List[CorpusRecord] = []
    
    def load(self) -> List[CorpusRecord]:
        """Load corpus from JSONL file and return structured records."""
        if not self.corpus_path.exists():
            logger.error(f"Corpus file not found: {self.corpus_path}")
            return []
        
        logger.info(f"Loading corpus from: {self.corpus_path}")
        records = []
        
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        record = self._validate_and_create_record(data, line_num)
                        if record:
                            records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(records)} valid records from corpus")
            self.records = records
            return records
            
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            return []
    
    def _validate_and_create_record(self, data: Dict[str, Any], line_num: int) -> Optional[CorpusRecord]:
        """Validate record data and create CorpusRecord."""
        required_fields = ["id", "source", "section", "text"]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.warning(f"Skipping line {line_num}: missing required field '{field}'")
                return None
        
        # Validate text content
        text = data["text"].strip()
        if not text:
            logger.warning(f"Skipping line {line_num}: empty text content")
            return None
        
        # Ensure skills is a list
        skills = data.get("skills", [])
        if not isinstance(skills, list):
            logger.warning(f"Skipping line {line_num}: skills must be a list")
            return None
        
        try:
            return CorpusRecord.from_dict(data)
        except Exception as e:
            logger.warning(f"Skipping line {line_num}: invalid record structure: {e}")
            return None
    
    def get_records_by_source(self, source: str) -> List[CorpusRecord]:
        """Get all records from a specific source."""
        return [record for record in self.records if record.source == source]
    
    def get_records_by_skills(self, skills: List[str]) -> List[CorpusRecord]:
        """Get records that contain any of the specified skills."""
        skill_set = set(skill.lower() for skill in skills)
        matching_records = []
        
        for record in self.records:
            record_skills = set(skill.lower() for skill in record.skills)
            if skill_set.intersection(record_skills):
                matching_records.append(record)
        
        return matching_records
    
    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        if not self.records:
            return {"total_records": 0}
        
        sources = {}
        skills_count = {}
        total_chars = 0
        
        for record in self.records:
            # Count by source
            sources[record.source] = sources.get(record.source, 0) + 1
            
            # Count skills
            for skill in record.skills:
                skills_count[skill] = skills_count.get(skill, 0) + 1
            
            # Count characters
            total_chars += len(record.text)
        
        return {
            "total_records": len(self.records),
            "sources": sources,
            "top_skills": sorted(skills_count.items(), key=lambda x: x[1], reverse=True)[:10],
            "total_characters": total_chars,
            "avg_chars_per_record": total_chars / len(self.records) if self.records else 0
        }


def load_corpus(corpus_path: str | Path) -> List[CorpusRecord]:
    """
    Convenience function to load corpus from file.
    
    Args:
        corpus_path: Path to corpus.jsonl file
        
    Returns:
        List of CorpusRecord objects
    """
    loader = CorpusLoader(corpus_path)
    return loader.load()


def main():
    """CLI test for data loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test corpus data loader")
    parser.add_argument("corpus_path", help="Path to corpus.jsonl file")
    parser.add_argument("--stats", action="store_true", help="Show corpus statistics")
    parser.add_argument("--source", help="Filter by source (resume/website/linkedin)")
    parser.add_argument("--skills", help="Comma-separated skills to filter by")
    
    args = parser.parse_args()
    
    # Load corpus
    records = load_corpus(args.corpus_path)
    
    if not records:
        print("❌ No records loaded")
        return
    
    print(f"✅ Loaded {len(records)} records")
    
    # Apply filters
    filtered_records = records
    
    if args.source:
        filtered_records = [r for r in filtered_records if r.source == args.source]
        print(f"📊 Filtered to {len(filtered_records)} records from source: {args.source}")
    
    if args.skills:
        skills_list = [s.strip() for s in args.skills.split(",")]
        skill_matches = []
        for record in filtered_records:
            record_skills = [s.lower() for s in record.skills]
            if any(skill.lower() in record_skills for skill in skills_list):
                skill_matches.append(record)
        filtered_records = skill_matches
        print(f"📊 Filtered to {len(filtered_records)} records with skills: {skills_list}")
    
    # Show stats
    if args.stats:
        loader = CorpusLoader(args.corpus_path)
        loader.records = records
        stats = loader.get_stats()
        
        print("\n📈 Corpus Statistics:")
        print(f"  Total records: {stats['total_records']}")
        print(f"  Total characters: {stats['total_characters']:,}")
        print(f"  Avg chars/record: {stats['avg_chars_per_record']:.1f}")
        
        print("\n📂 By Source:")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count} records")
        
        print("\n🏷️  Top Skills:")
        for skill, count in stats['top_skills']:
            print(f"  {skill}: {count} records")
    
    # Show sample records
    print(f"\n📄 Sample Records (showing first 3):")
    for i, record in enumerate(filtered_records[:3], 1):
        print(f"\n{i}. {record.id}")
        print(f"   Source: {record.source}")
        print(f"   Section: {record.section}")
        print(f"   Skills: {', '.join(record.skills[:5])}{'...' if len(record.skills) > 5 else ''}")
        print(f"   Text: {record.text[:100]}...")
        if record.url:
            print(f"   URL: {record.url}")


if __name__ == "__main__":
    main()
