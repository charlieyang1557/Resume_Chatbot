# 📚 Resume Chatbot Corpus Enhancement Guide

This guide shows how to process your personal website and LinkedIn profile data to enhance your resume chatbot corpus with the existing `make_corpus_from_text.py` script.

## 🎯 Overview

Your `make_corpus_from_text.py` script is already perfectly designed for this task! It:
- ✅ Chunks text into 800-1200 character segments with 120 character overlap
- ✅ Auto-infers skills from 40+ technical keywords (Python, SQL, AWS, etc.)
- ✅ Generates JSONL records with proper structure: `{id, source, section, date_range, skills, text, url}`
- ✅ Supports both single-file and batch processing via manifest
- ✅ Creates unique IDs like `website#about_me_c1`, `linkedin#summary_c2`

## 🚀 Quick Start

### 1. Single File Processing

Process your website's "About" section:
```bash
python make_corpus_from_text.py \
  --text_file about.txt \
  --source website \
  --section "About Me" \
  --url "https://charlieyang1557.github.io/aboutme/"
```

Process LinkedIn summary:
```bash
python make_corpus_from_text.py \
  --text_file linkedin_summary.txt \
  --source linkedin \
  --section "Summary" \
  --url "https://www.linkedin.com/in/yutianyang/"
```

### 2. Batch Processing with Manifest

Create `manifest.yaml`:
```yaml
- text_file: "about.txt"
  source: "website"
  section: "About Me"
  url: "https://charlieyang1557.github.io/aboutme/"
  date_range: ""
  id_prefix: "website#about"

- text_file: "projects.txt"
  source: "website"
  section: "Projects"
  url: "https://charlieyang1557.github.io/aboutme/"
  date_range: ""
  id_prefix: "website#projects"

- text_file: "linkedin_summary.txt"
  source: "linkedin"
  section: "Summary"
  url: "https://www.linkedin.com/in/yutianyang/"
  date_range: ""
  id_prefix: "linkedin#summary"

- text_file: "linkedin_experience.txt"
  source: "linkedin"
  section: "Experience"
  url: "https://www.linkedin.com/in/yutianyang/"
  date_range: ""
  id_prefix: "linkedin#experience"
```

Run batch processing:
```bash
python make_corpus_from_text.py --manifest manifest.yaml
```

## 📋 Step-by-Step Instructions

### Step 1: Prepare Your Text Files

#### Option A: Manual Export
1. Copy your website "About" section → save as `about.txt`
2. Copy your website "Projects" section → save as `projects.txt`
3. Copy your LinkedIn "Summary" → save as `linkedin_summary.txt`
4. Copy your LinkedIn "Experience" descriptions → save as `linkedin_experience.txt`

#### Option B: Automated Scraping (Optional)
Use the helper script below to scrape your website automatically:

```python
# scrape_website.py
import trafilatura
import requests
from pathlib import Path

def scrape_website_sections(url: str, output_dir: str = "."):
    """Scrape website and extract sections into separate files."""
    
    # Fetch the webpage
    response = requests.get(url)
    response.raise_for_status()
    
    # Extract main content
    main_content = trafilatura.extract(response.text)
    if not main_content:
        print(f"⚠️  Could not extract content from {url}")
        return
    
    # Save full content for manual sectioning
    Path(output_dir).mkdir(exist_ok=True)
    with open(f"{output_dir}/website_full.txt", "w", encoding="utf-8") as f:
        f.write(main_content)
    
    print(f"✅ Scraped content saved to {output_dir}/website_full.txt")
    print("📝 Please manually extract sections into separate files:")
    print("   - about.txt (About Me section)")
    print("   - projects.txt (Projects section)")
    print("   - experience.txt (Work Experience section)")

if __name__ == "__main__":
    # Scrape your personal website
    scrape_website_sections("https://charlieyang1557.github.io/aboutme/", "website_data/")
```

Run the scraper:
```bash
pip install trafilatura requests
python scrape_website.py
```

### Step 2: Create Text Files

After scraping or manual export, you should have files like:
```
website_data/
├── about.txt          # About Me section
├── projects.txt       # Projects section
└── linkedin_data/
    ├── linkedin_summary.txt
    └── linkedin_experience.txt
```

### Step 3: Process with Manifest

Create `manifest.yaml`:
```yaml
# Website sections
- text_file: "website_data/about.txt"
  source: "website"
  section: "About Me"
  url: "https://charlieyang1557.github.io/aboutme/"
  date_range: ""
  id_prefix: "website#about"

- text_file: "website_data/projects.txt"
  source: "website"
  section: "Projects"
  url: "https://charlieyang1557.github.io/aboutme/"
  date_range: ""
  id_prefix: "website#projects"

# LinkedIn sections
- text_file: "website_data/linkedin_data/linkedin_summary.txt"
  source: "linkedin"
  section: "Summary"
  url: "https://www.linkedin.com/in/yutianyang/"
  date_range: ""
  id_prefix: "linkedin#summary"

- text_file: "website_data/linkedin_data/linkedin_experience.txt"
  source: "linkedin"
  section: "Experience"
  url: "https://www.linkedin.com/in/yutianyang/"
  date_range: ""
  id_prefix: "linkedin#experience"
```

### Step 4: Generate Corpus Records

```bash
python make_corpus_from_text.py --manifest manifest.yaml --out corpus_extra.jsonl
```

Expected output:
```
Wrote 12 records to corpus_extra.jsonl
```

### Step 5: Merge with Main Corpus

#### Option A: Simple Append (Recommended)
```bash
# Backup your original corpus
cp corpus.jsonl corpus.jsonl.backup

# Append new records to main corpus
cat corpus_extra.jsonl >> corpus.jsonl

# Verify the merge
echo "Original corpus lines: $(wc -l < corpus.jsonl.backup)"
echo "New records: $(wc -l < corpus_extra.jsonl)"
echo "Total lines: $(wc -l < corpus.jsonl)"
```

#### Option B: Python Merge (Safer)
```python
# merge_corpus.py
import json
from pathlib import Path

def merge_corpus_files(main_file: str, extra_file: str, output_file: str = None):
    """Safely merge corpus files with duplicate checking."""
    
    # Load main corpus
    main_records = []
    with open(main_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                main_records.append(json.loads(line))
    
    # Load extra records
    extra_records = []
    with open(extra_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                extra_records.append(json.loads(line))
    
    # Check for ID conflicts
    main_ids = {r['id'] for r in main_records}
    conflicts = [r['id'] for r in extra_records if r['id'] in main_ids]
    
    if conflicts:
        print(f"⚠️  ID conflicts found: {conflicts}")
        print("Consider updating ID prefixes in manifest.yaml")
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
    return True

if __name__ == "__main__":
    merge_corpus_files("corpus.jsonl", "corpus_extra.jsonl")
```

Run the Python merge:
```bash
python merge_corpus.py
```

## 🔧 Advanced Usage

### Custom Chunking Parameters
```bash
python make_corpus_from_text.py \
  --manifest manifest.yaml \
  --min_len 600 \
  --max_len 1000 \
  --overlap 100 \
  --out corpus_extra.jsonl
```

### Processing Multiple Sources
```bash
# Process website and LinkedIn separately
python make_corpus_from_text.py \
  --text_file website_data/about.txt \
  --source website \
  --section "About Me" \
  --url "https://charlieyang1557.github.io/aboutme/" \
  --out website_corpus.jsonl

python make_corpus_from_text.py \
  --text_file linkedin_summary.txt \
  --source linkedin \
  --section "Summary" \
  --url "https://www.linkedin.com/in/yutianyang/" \
  --out linkedin_corpus.jsonl

# Merge both
cat website_corpus.jsonl linkedin_corpus.jsonl > corpus_extra.jsonl
```

### Custom ID Prefixes
```yaml
- text_file: "about.txt"
  source: "website"
  section: "About Me"
  url: "https://charlieyang1557.github.io/aboutme/"
  id_prefix: "personal#about"  # Custom prefix
```

This generates IDs like: `personal#about_c1`, `personal#about_c2`, etc.

## 📊 Expected Output

Your `corpus_extra.jsonl` will contain records like:

```json
{"id": "website#about_c1", "source": "website", "section": "About Me", "date_range": "", "skills": ["Python", "SQL", "AWS"], "text": "I'm a passionate data scientist with expertise in machine learning and analytics. I love building end-to-end ML pipelines using Python, SQL, and cloud platforms like AWS...", "url": "https://charlieyang1557.github.io/aboutme/"}
{"id": "website#about_c2", "source": "website", "section": "About Me", "date_range": "", "skills": ["TensorFlow", "Time Series"], "text": "My recent projects include developing time series forecasting models with TensorFlow and building real-time analytics dashboards. I'm particularly interested in anomaly detection...", "url": "https://charlieyang1557.github.io/aboutme/"}
{"id": "linkedin#summary_c1", "source": "linkedin", "section": "Summary", "date_range": "", "skills": ["Python", "SQL", "Tableau"], "text": "Experienced Data Scientist with 3+ years of experience in machine learning, statistical analysis, and data visualization. Proficient in Python, SQL, and Tableau for extracting actionable insights...", "url": "https://www.linkedin.com/in/yutianyang/"}
```

## 🔍 Quality Check

After processing, verify your corpus:

```bash
# Check record count
echo "Total records: $(wc -l < corpus.jsonl)"

# Check for duplicate IDs
python -c "
import json
ids = []
with open('corpus.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            ids.append(json.loads(line)['id'])
duplicates = [id for id in set(ids) if ids.count(id) > 1]
print(f'Duplicate IDs: {duplicates if duplicates else \"None\"}')
print(f'Unique records: {len(set(ids))}')
"

# Sample a few records
head -3 corpus.jsonl | python -m json.tool
```

## 🔄 Rebuild FAISS Index

After merging your enhanced corpus, rebuild the FAISS index:

```bash
# If using your enhanced chatbot
python -c "
from resume_chatbot.enhanced_chatbot import create_enhanced_chatbot
from pathlib import Path

chatbot = create_enhanced_chatbot(
    resume_directory=Path('data/resume'),
    llm_backend='ollama',
    use_faiss=True
)
chatbot.rebuild_index()
print('✅ FAISS index rebuilt with enhanced corpus')
"
```

Or if using the original system:
```bash
# Rebuild index with your existing index_builder.py
python index_builder.py --corpus corpus.jsonl --output data/faiss_index
```

## 🎯 Pro Tips

### 1. Optimize Chunking
- **Short sections** (like contact info): Use smaller chunks (400-600 chars)
- **Long sections** (like detailed project descriptions): Use larger chunks (1000-1200 chars)
- **Technical content**: Increase overlap to 150-200 chars for better context

### 2. Enhance Skill Detection
Your script already detects 40+ skills, but you can extend `SKILL_KEYWORDS` in the script:
```python
SKILL_KEYWORDS.update({
    "kubernetes": "Kubernetes",
    "docker": "Docker", 
    "airflow": "Airflow",
    "spark": "Spark",
    "kafka": "Kafka",
    "redis": "Redis",
    "mongodb": "MongoDB",
    "postgresql": "PostgreSQL"
})
```

### 3. Organize by Date
Add date ranges to your manifest for better temporal context:
```yaml
- text_file: "projects_2023.txt"
  source: "website"
  section: "Recent Projects"
  date_range: "2023-2024"
  url: "https://charlieyang1557.github.io/aboutme/"
```

### 4. Batch Processing Workflow
```bash
#!/bin/bash
# process_all.sh - Complete workflow

echo "🔄 Processing website and LinkedIn data..."

# Scrape website (optional)
python scrape_website.py

# Process with manifest
python make_corpus_from_text.py --manifest manifest.yaml --out corpus_extra.jsonl

# Merge with main corpus
python merge_corpus.py

# Rebuild FAISS index
python -c "
from resume_chatbot.enhanced_chatbot import create_enhanced_chatbot
from pathlib import Path
chatbot = create_enhanced_chatbot(resume_directory=Path('data/resume'))
chatbot.rebuild_index()
print('✅ Complete! Corpus enhanced and index rebuilt.')
"

echo "🎉 Enhancement complete!"
```

## 🚨 Troubleshooting

### Common Issues

**"No input provided" error:**
```bash
# Check your manifest syntax
python -c "import yaml; print(yaml.safe_load(open('manifest.yaml')))"
```

**ID conflicts during merge:**
```bash
# Check for duplicate IDs
grep -o '"id": "[^"]*"' corpus.jsonl | sort | uniq -d
```

**Empty chunks:**
```bash
# Check text file content
wc -w about.txt  # Should have > 100 words
```

**Skills not detected:**
- Check if your text contains the keywords (case-insensitive)
- Extend `SKILL_KEYWORDS` dictionary in the script

## 📝 Final Checklist

- [ ] ✅ Text files created (about.txt, projects.txt, linkedin_*.txt)
- [ ] ✅ Manifest.yaml configured with all sections
- [ ] ✅ `corpus_extra.jsonl` generated successfully
- [ ] ✅ Corpus merged without ID conflicts
- [ ] ✅ FAISS index rebuilt with enhanced corpus
- [ ] ✅ Chatbot tested with new data

---

**🎉 Your resume chatbot now has comprehensive coverage of your website and LinkedIn profile!**

The enhanced corpus will provide much richer context for answering questions about your background, projects, and experience. Your chatbot can now reference specific projects from your website, detailed LinkedIn experience descriptions, and personal insights from your About section.
