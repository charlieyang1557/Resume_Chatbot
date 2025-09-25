#!/usr/bin/env python3
"""
Simple test for data loader without heavy dependencies.
"""

import json
from pathlib import Path
from data_loader import load_corpus, CorpusLoader


def main():
    """Test data loader functionality."""
    corpus_path = "corpus_original.jsonl"
    
    print("🧪 Testing Data Loader")
    print("=" * 40)
    
    # Test loading
    print(f"📚 Loading corpus from: {corpus_path}")
    records = load_corpus(corpus_path)
    
    if not records:
        print("❌ No records loaded")
        return
    
    print(f"✅ Loaded {len(records)} records")
    
    # Test loader stats
    loader = CorpusLoader(corpus_path)
    loader.records = records
    stats = loader.get_stats()
    
    print(f"\n📊 Corpus Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Total characters: {stats['total_characters']:,}")
    print(f"  Avg chars/record: {stats['avg_chars_per_record']:.1f}")
    
    print(f"\n📂 By Source:")
    for source, count in stats['sources'].items():
        print(f"  {source}: {count} records")
    
    print(f"\n🏷️  Top Skills:")
    for skill, count in stats['top_skills']:
        print(f"  {skill}: {count} records")
    
    # Test filtering
    print(f"\n🔍 Testing Filters:")
    
    # By source
    resume_records = loader.get_records_by_source("resume")
    print(f"  Resume records: {len(resume_records)}")
    
    website_records = loader.get_records_by_source("website")
    print(f"  Website records: {len(website_records)}")
    
    # By skills
    python_records = loader.get_records_by_skills(["Python"])
    print(f"  Records with Python: {len(python_records)}")
    
    sql_records = loader.get_records_by_skills(["SQL"])
    print(f"  Records with SQL: {len(sql_records)}")
    
    # Show sample records
    print(f"\n📄 Sample Records:")
    for i, record in enumerate(records[:3], 1):
        print(f"\n{i}. {record.id}")
        print(f"   Source: {record.source}")
        print(f"   Section: {record.section}")
        print(f"   Skills: {', '.join(record.skills[:5])}{'...' if len(record.skills) > 5 else ''}")
        print(f"   Text: {record.text[:100]}...")
        if record.url:
            print(f"   URL: {record.url}")
    
    # Test PTC Onshape question
    print(f"\n🎯 Testing PTC Onshape Query:")
    ptc_records = [r for r in records if "PTC" in r.text or "Onshape" in r.text]
    print(f"  Found {len(ptc_records)} records mentioning PTC/Onshape")
    
    for record in ptc_records:
        print(f"    - {record.id}: {record.section}")
        print(f"      Skills: {', '.join(record.skills[:5])}")
        print(f"      Text: {record.text[:150]}...")
    
    print(f"\n✅ Data loader test completed successfully!")


if __name__ == "__main__":
    main()
