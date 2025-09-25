#!/usr/bin/env python3
"""
Safe corpus merging script with duplicate ID detection.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_corpus(file_path: str) -> List[Dict[str, Any]]:
    """Load corpus from JSONL file."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON error in {file_path} line {line_num}: {e}")
    return records


def merge_corpus_files(
    main_file: str, 
    extra_file: str, 
    output_file: str = None,
    check_conflicts: bool = True
) -> bool:
    """Safely merge corpus files with duplicate checking."""
    
    print(f"📚 Loading main corpus: {main_file}")
    main_records = load_corpus(main_file)
    
    print(f"📚 Loading extra corpus: {extra_file}")
    extra_records = load_corpus(extra_file)
    
    print(f"📊 Main corpus: {len(main_records)} records")
    print(f"📊 Extra corpus: {len(extra_records)} records")
    
    if check_conflicts:
        # Check for ID conflicts
        main_ids = {r['id'] for r in main_records}
        conflicts = [r['id'] for r in extra_records if r['id'] in main_ids]
        
        if conflicts:
            print(f"⚠️  ID conflicts found: {conflicts}")
            print("💡 Consider updating ID prefixes in your manifest")
            response = input("Continue anyway? (y/N): ").lower().strip()
            if response != 'y':
                print("❌ Merge cancelled")
                return False
    
    # Merge records
    merged_records = main_records + extra_records
    
    # Write merged corpus
    output_path = output_file or main_file
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in merged_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Merged {len(main_records)} + {len(extra_records)} = {len(merged_records)} records")
    print(f"📁 Saved to: {output_path}")
    
    # Verify merge
    unique_ids = len(set(r['id'] for r in merged_records))
    print(f"🔍 Unique IDs: {unique_ids} (should equal total records)")
    
    return True


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safely merge corpus JSONL files")
    parser.add_argument("main_file", help="Main corpus file (will be updated)")
    parser.add_argument("extra_file", help="Extra corpus file to merge")
    parser.add_argument("--output", "-o", help="Output file (default: update main_file)")
    parser.add_argument("--force", "-f", action="store_true", help="Skip conflict checking")
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.main_file).exists():
        print(f"❌ Main file not found: {args.main_file}")
        sys.exit(1)
    
    if not Path(args.extra_file).exists():
        print(f"❌ Extra file not found: {args.extra_file}")
        sys.exit(1)
    
    # Merge files
    success = merge_corpus_files(
        args.main_file, 
        args.extra_file, 
        args.output,
        check_conflicts=not args.force
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
